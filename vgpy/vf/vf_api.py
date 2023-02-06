"""
 modified by Jack 2021/03/04
"""
# from main import app
from .config import config #Jack add 把讀取 JSON 檔移到另一個檔案
import time
import os
import numpy as np
import json
import threading
from flask import Blueprint,render_template, request
from multiprocessing import Process
from collections import OrderedDict
from . import vf_backend as vfbackend
from .web_controller import WebController
from ..global_object import StreamingQueues, GlobalVar
from ..yolov5.vf_yolov5_detect import Detector
from ..utils import json_utils
from vgpy.utils.logger import create_logger

app_vf = Blueprint('app_vf', __name__)
app_obj = Blueprint('app_obj', __name__)

# 全域變數
config_dir = config.CONFIG_DIR

VfVar = vfbackend.VfVar
logger = create_logger()


class VFMain:
    def __init__(self, q:StreamingQueues, vfvar:vfbackend.VfVar, globalvar:GlobalVar):
        self.vfvar = vfvar
        self.globalvar = globalvar
        self.web_controller = WebController()
        self.web_controller.q = q
        self.streaming_queues = q

    
    def vf_app_init(self):
        vfvar = self.vfvar
        globalvar = self.globalvar
        web_controller = self.web_controller
        q = self.streaming_queues

        @app_vf.route('/dist_detect/index', methods=['GET', 'POST'])  # 主页
        def distance_index():
            current_camid = int(globalvar.cam_id)
            cam_num = int(globalvar.cam_num)
            vfvar.vf_alarm_set = config.get_alarm_setting('vf')
            # 沒value 填入None，防止前端 jinja2 取 dict no key error 
            alarm_setting = {
                "channel": vfvar.vf_alarm_set.get('channel'),
                "level": vfvar.vf_alarm_set.get('level'),
                "log": vfvar.vf_alarm_set.get('log')
            }
            vfvar.obj_detect_mode = False
            web_controller.obj_detect_mode = False  # Wen更新 -以webcontroller(頁面上)控管兩種模式的切換
            web_controller.shot = 0
            q.c_queue.put((VfVar.SIGNAL.CHANGE_DETECT_MODE, vfvar))
                                    
            bili_data = config.get_bili(web_controller.obj_detect_mode)
            data_table = eval(config.get_data_table()) if config.get_data_table() !="" else config.get_data_table()
            a_coord = config.get_canvas_area_str(web_controller.obj_detect_mode)
            mail_group = globalvar.mail_group   #vfbackend.get_mail_group()
            vfvar.current_model_weights, vfvar.current_model_class = config.get_model_config(vfvar.obj_detect_mode)
            current_weights = vfvar.current_model_weights.replace('.pt','')
            current_class = vfvar.current_model_class
            weights_list = vfbackend.get_model_weights_list()
            class_list = vfbackend.get_model_class_list(current_weights)
            alarm_log_json =vfbackend.get_alarm_log_json()
            alarm_log_json = OrderedDict(sorted(alarm_log_json['vf'].items(), key= lambda x: x[0], reverse=True))

            logger.info('測距/物件:進入測距監控系統')
            if request.method =='POST':
                vfvar.current_model_class = select_class =  request.values.getlist("class_list")
                select_weights = request.values.get("scene_list")
                vfvar.current_model_weights = select_weights+'.pt'
                class_list = vfbackend.get_model_class_list(select_weights)
                mode = "obj" if vfvar.obj_detect_mode else "vf"
                config.save_model_config(mode, vfvar.current_model_weights, select_class) #儲存模型配置
                q.c_queue.put((VfVar.SIGNAL.CHANGE_MODEL_WEIGHTS, vfvar))
                current_weights = select_weights
                current_class = select_class
                logger.info(f"change yolov5 weights and class to {select_weights} and {select_class}")
                    
            return render_template(
                'vf_danger_area.html', data_dict=data_table, weights=current_weights, class_list=class_list,select_class=current_class,
                weights_list=weights_list, bili_data=bili_data, area_Coord=a_coord,cam_data=(current_camid,cam_num),\
                alarm_set=alarm_setting, mail_group_dict=mail_group, alarm_log=alarm_log_json)
                           

        @app_vf.route('/show_img/<string:target_dir>/<string:filename>', methods=['GET'])
        def show_photo(target_dir, filename):
            return vfbackend.return_image_to_web(target_dir,filename)

        @app_vf.route("/Distance/saveBili",methods=['POST',])
        def saveBili():
            global canvas_width, canvas_height
            ds = request.get_json()
            bili_data = json.loads(ds)
            bili = bili_data['bili']
            bili_coord = bili_data['coord']
            
            canvas_width = bili_data['img_width']
            canvas_height = bili_data['img_height']
            
            if bili is not None and bili_coord is not None:
                config.save_bili(web_controller.obj_detect_mode, bili_data)
                # config.save_bili_coord(web_controller.obj_detect_mode, bili_coord)
                print('save new bili:', bili)
                print('save new bili_coord:', bili_coord)
                vfbackend.save_past_setting_bili_img(web_controller.obj_detect_mode, web_controller.img_video_feed)
                logger.info('測距/物件:比例尺設定成功')
                return "儲存比例尺成功"
            else:
                logger.error('測距/物件:比例尺設定失敗')
                return "錯誤，請聯絡窗口"


        @app_vf.route("/Distance/saveAreaCoord",methods=['POST','GET'])
        def saveAreaCoord():
            global canvas_width, canvas_height
            if request.method =='POST':
                area_json = request.get_json()
                canvas_areas = area_json['area_coord'] # 前端的 canvas 上畫的圍籬座標
                canvas_height =int(area_json['img_height'])
                canvas_width = int(area_json['img_width'])
                try:                    
                    w,h = globalvar.cam_wh
                    print('寬*高'+ str(w) +','+str(h))
                    
                    cam_areas = config.from_canvas_to_camera(
                        canvas_areas, canvas_wh=(canvas_width, canvas_height), cam_wh=(w, h))
                    
                    config.save_fence_areas_ponits(web_controller.obj_detect_mode, canvas_data=(canvas_areas,canvas_width,canvas_height), cam_areas=cam_areas)  #add 儲存canvas所有前端資訊          
                    print('新設定的圍籬區域:\n', cam_areas) # [[('1781', '41'), ('1670', '127'), ('1858', '129')]]
                    vfbackend.save_past_setting_area_img(web_controller.obj_detect_mode, web_controller.img_video_feed)

                    ## Wen新增 =================================================#
                    if web_controller.obj_detect_mode:
                        vfvar.cam_obj_areas = cam_areas
                        q.c_queue.put((VfVar.SIGNAL.CHANGE_OBJ_AREAS, vfvar))
                    else:
                        vfvar.cam_vf_areas = cam_areas
                        q.c_queue.put((VfVar.SIGNAL.CHANGE_VF_AREAS, vfvar))
                    #===========================================================#
                    logger.info('測距/物件:虛擬圍籬設定成功')
                    return '圍籬區域儲存成功!'

                except Exception as e:
                    logger.error(f'測距/物件:虛擬圍籬設定失敗, {e}')
                    return '圍籬區域儲存失敗'

        @app_vf.route("/Distance/saveAlarmSetting",methods=['POST'])
        def saveAlarmSetting():
            if request.method =='POST':
                w_json = request.get_json()
                mode = w_json['mode']
                # mode, items, data = w_json['mode'], w_json['item'], w_json['data']
                try:
                    for item in w_json['items']:
                        data = w_json['items'][item]
                        save_json = config.save_alarm_setting(mode, item, data)
                        if mode == 'vf':
                            vfvar.vf_alarm_set[item] = save_json
                        else:
                            vfvar.obj_alarm_set[item] = save_json
                    if mode == 'vf':
                        q.c_queue.put((VfVar.SIGNAL.CHANGE_VF_ALARM_SET, vfvar))
                    elif mode == 'obj':
                        q.c_queue.put((VfVar.SIGNAL.CHANGE_OBJ_ALARM_SET, vfvar))
                    logger.info('測距/物件:異常通報設定成功')
                    return '異常通報設定 儲存成功'
                except Exception as e :
                    logger.error(f'測距/物件:異常通報設定失敗, {e}')
                    return '異常通報設定 儲存失敗'

        @app_vf.route("/Distance/alarm_clear",methods=['POST'])
        def alarm_clear():
            if request.method =='POST':
                q.c_queue.put(VfVar.SIGNAL.ALARM_CLEAR)
                logger.info('測距/物件:手動清除異常警報')
                return '已清除警報！！'

        @app_vf.route("/Distance/selectListChange",methods=['POST','GET'])
        def selectListChange():
            if request.method =='POST':
                select_weights = request.get_json()
                print(select_weights)
                if select_weights is not None:
                    class_list = vfbackend.get_model_class_list(select_weights)
                    return json.dumps(class_list,ensure_ascii=False)
                 
        @app_vf.route("/Distance/SettingPageOn",methods=['POST',])
        def SettingPageOn():
            web_controller.setTab = int(request.get_json()) # 設定頁打開時，預設傳入值為1，代表 虛擬圍籬設定中
            web_controller.setFlag = 1
            # web_controller.shot = 0
            q.c_queue.put(VfVar.SIGNAL.SETTING_PAGE_ON)
            print('Setting Page ON,shot='+str(web_controller.shot))
            return '設定頁開啟'

        @app_vf.route("/Distance/MainPageOn",methods=['POST'])
        def MainPageOn():
            web_controller.setFlag = 0
            print('Main Page ON')
            return '功能主頁開啟'

        @app_vf.route("/Distance/AlarmLogPageOn",methods=['POST'])
        def AlarmLogPageOn():
            web_controller.setTab = request.get_json() # 設定頁打開時，預設傳入值為1，代表 虛擬圍籬設定中
            web_controller.setFlag = 1
            web_controller.shot = 0
            q.c_queue.put(VfVar.SIGNAL.SETTING_PAGE_ON) # 為中斷辨識
            log_json = vfbackend.get_alarm_log_json()
            if vfvar.obj_detect_mode:
                log_json = log_json['obj'] 
            else:
                log_json = log_json['vf']
            print('AlarmLog Page ON,shot='+str(web_controller.shot))
            return log_json

        @app_vf.route("/Distance/screenShot",methods=['POST','GET'])
        def ds_screenShot():
            web_controller.setFlag = 1
            if web_controller.shot == 0:
                web_controller.shot = 1
                if web_controller.setTab == 1: #area
                    vfbackend.save_past_setting_area_img(web_controller.obj_detect_mode, web_controller.img_video_feed)
                elif web_controller.setTab == 2: #bili
                    vfbackend.save_past_setting_bili_img(web_controller.obj_detect_mode, web_controller.img_video_feed)
                logger.info('測距/物件:儲存截圖')                
            else:
                web_controller.shot = 0
            print('---- screenshot ----,shot='+str(web_controller.shot))
            return str(web_controller.shot)


        @app_vf.route("/Distance/getHistoryPic",methods=['POST','GET'])
        def getHistoryPic():
            web_controller.setFlag = 1
            if request.method =='POST':
                web_controller.history = request.get_json()
                if web_controller.history == 1:
                    web_controller.shot = 2
                    print('取得過往設定照片,history='+str(web_controller.history)+',setTab='+str(web_controller.setTab)+',shot='+str(web_controller.shot))
                else:
                    web_controller.shot = 0
                    print('取得現在設定照片,history='+str(web_controller.history)+',setTab='+str(web_controller.setTab)+',shot='+str(web_controller.shot))
                
                logger.info('測距/物件:切換過往/即時影像')
                return '影像切換成功'


        @app_vf.route("/Distance/SettingTabsSwitch",methods=['POST','GET'])
        def SettingTabsSwitch():
            if request.method =='POST':
                web_controller.setTab = int(request.get_json())
                web_controller.setFlag = 1
                # web_controller.shot = 0
                web_controller.history = 0
                print('setTab='+ str(web_controller.setTab))
                return 'success'

        @app_vf.route('/Distance/info_Update', methods=['POST',])
        def info_Update():
            data_table_str = request.get_json()
            config.save_data_table(data_table_str)  
            logger.info('測距/物件:儲存DataTable資訊')
            return "更新成功"

        return app_vf


    def obj_app_init(self):
        vfvar = self.vfvar
        web_controller = self.web_controller
        q = self.streaming_queues
        globalvar = self.globalvar
                                
        @app_obj.route('/object_detect/index', methods=['GET', 'POST'])  # 主页
        def object_index():
            current_camid = globalvar.cam_id
            cam_num = int(globalvar.cam_num)
            vfvar.obj_detect_mode = True
            web_controller.obj_detect_mode = True  # Wen更新 -以webcontroller(頁面上)控管兩種模式的切換
            web_controller.shot = 0
            q.c_queue.put((VfVar.SIGNAL.CHANGE_DETECT_MODE, vfvar))

            vfvar.obj_alarm_set = config.get_alarm_setting('obj')
            # 沒value 填入None，防止前端 jinja2 取 dict no key error 
            alarm_setting = {
                "channel": vfvar.obj_alarm_set.get('channel'),
                "alarm_threhold": vfvar.obj_alarm_set.get('alarm_threhold'),
                "log": vfvar.obj_alarm_set.get('log')
            }
            mail_group = globalvar.mail_group
            bili_data = config.get_bili(web_controller.obj_detect_mode)
            data_table = config.get_data_table()
            a_coord = config.get_canvas_area_str(web_controller.obj_detect_mode)
            # b_coord = config.get_bili_coord(web_controller.obj_detect_mode)
            vfvar.current_model_weights, vfvar.current_model_class = config.get_model_config(vfvar.obj_detect_mode)
            current_weights = vfvar.current_model_weights.replace('.pt','')
            current_class = vfvar.current_model_class
            weights_list = vfbackend.get_model_weights_list()
            class_list = vfbackend.get_model_class_list(current_weights)
            alarm_log_json =vfbackend.get_alarm_log_json()
            alarm_log_json = OrderedDict(sorted(alarm_log_json['obj'].items(), key= lambda x: x[0], reverse=True))
            logger.info('測距/物件:進入物件監控系統')

            if request.method =='POST':
                vfvar.current_model_class = select_class =  request.values.getlist("class_list")
                select_weights = request.values.get("scene_list")
                vfvar.current_model_weights = select_weights+'.pt'
                class_list = vfbackend.get_model_class_list(select_weights)
                # print(select_weights,select_class) #如select_class為空，則以預設rm_class篩選類別
                mode = "obj" if vfvar.obj_detect_mode else "vf"
                config.save_model_config(mode, vfvar.current_model_weights, select_class) #儲存模型配置
                q.c_queue.put((VfVar.SIGNAL.CHANGE_MODEL_WEIGHTS, vfvar))
                current_class = select_class
                current_weights = select_weights
                logger.info(f"change yolov5 weights and class to {select_weights} and {select_class}")
             
            return render_template(
                'vf_object_area.html', data_dict=eval(data_table), weights=current_weights, class_list=class_list,select_class=current_class,
                weights_list=weights_list, bili_data=bili_data, area_Coord=a_coord,cam_data=(current_camid,cam_num)
                ,alarm_set=alarm_setting,mail_group_dict=mail_group, alarm_log=alarm_log_json)
                 

        return app_obj

    def app_vf_start(self):
        # 開始虛擬圍籬 距離偵測
        current_process = Process(
            target=vfbackend.vf_detect_init,
                args=(self.streaming_queues, self.vfvar, self.globalvar), daemon=False)
        current_process.start()
        logger.info('測距/物件: subprocess啟動')
        self.streaming_queues.c_queue.get() # 卡在這等 subprocess 內的模型讀取完成        
        return current_process

    def get_img(self, current_frame):
        return self.web_controller.get_img(current_frame)



if __name__ == '__main__':
    pass

