import os
from pickle import GLOBAL
import cv2
import json
import yaml
from ..utils.json_utils import save_json, load_json
from .config import config #Jack add 把讀取 JSON 檔移到另一個檔案
from ..global_function import process_response_image
from vgpy.utils.logger import create_logger
logger = create_logger()

import time

current_dir = os.path.dirname(os.path.abspath(__file__))
GLOBAL_CONFIG_DIR = os.path.join(os.getcwd(),'vgpy', 'config')
CONFIG_DIR = os.path.join(current_dir, 'config')
IMG_DIR = os.path.join(current_dir, 'img')
PAST_AREA_IMG = os.path.join(IMG_DIR, 'setting_AreaPic.jpg')
PAST_BILI_IMG = os.path.join(IMG_DIR, 'setting_BiliPic.jpg')
VF_PAST_AREA_IMG = os.path.join(IMG_DIR, 'setting_vf_AreaPic.jpg')
VF_PAST_BILI_IMG = os.path.join(IMG_DIR, 'setting_vf_BiliPic.jpg')
OBJ_PAST_AREA_IMG = os.path.join(IMG_DIR, 'setting_obj_AreaPic.jpg')
OBJ_PAST_BILI_IMG = os.path.join(IMG_DIR, 'setting_obj_BiliPic.jpg')
ALARM_IMG = os.path.join(IMG_DIR, 'Alarm_Pic.jpg')
AUDIO_DIR = os.path.join(current_dir, 'audio')
# ALARM_MP3_PATH = os.path.join(AUDIO_DIR, 'alarm.mp3')
MODEL_WEIGHTS_DIR = os.path.join(os.getcwd(),'vgpy','yolov5','weights')

HISTORY_MP3_JSON_FILENAME = 'history.json'
HISTORY_MP3_JSON_PATH = os.path.join(AUDIO_DIR, HISTORY_MP3_JSON_FILENAME)

from enum import Enum, auto
# 虛擬圍籬的 process 會用到的狀態等
class VfVar:
    class SIGNAL(Enum):
        """
             @attr
                CHANGE_DETECT_MODE: 更改虛擬圍籬的模式，圍籬模式 或 物件模式
                CHANGE_(MODE)_AREAS: 當圍籬設定有改變的時候，如新增虛擬圍籬區域 或 刪除虛擬圍籬區域
                SETTING_PAGE_ON: 當設定頁面打開時，傳入此訊號讓 detector 的 alarm flag 變成0，這樣才不會在設定頁面時，還一直發警報
                CHANGE_ALARM_SET: 更改(各等級)警報安全距離
                ALARM_CLEAR: 警報清除
        """
        CHANGE_DETECT_MODE = auto()
        # CHANGE_AREAS = auto()
        ## Wen 新增 =====================#
        CHANGE_VF_AREAS = auto()
        CHANGE_OBJ_AREAS = auto()
        CHANGE_VF_ALARM_SET = auto()
        CHANGE_OBJ_ALARM_SET = auto()
        # CHANGE_ALARM_CHANNEL = auto()
        CHANGE_MODEL_WEIGHTS = auto()
        ALARM_CLEAR = auto()
        #================================#
        CHANGE_SAVE_DISTANCE = auto() #TODO 尚未實作從前端更改安全距離
        SETTING_PAGE_ON = auto()
        
    def __init__(self):
        """
        @attr
            obj_detect_mode: 目前模式 共兩種，分別為圍籬／物件模式 (預設為：虛擬圍籬)
            cam_areas:  (拆分成兩種模式儲存後 已廢棄)
            least_safe_distance: 最長偵測距離 低於此距離才會偵測距離
            cam_vf_areas: 已轉換成 Webcam長寬比的圍籬區域座標 for 圍籬模式
            cam_obj_areas: 已轉換成 Webcam長寬比的圍籬區域座標 for 物件模式
            alarm_level_set: 人物靠近的等級警報距離
            alarm_channel: 要警告發報的頻道與群組(ex:Email、LINE)
            current_model_weights: 目前的模型權重，當 切換模型 時會用到
            current_model_class: 目前模型的類別，前端可 篩選 要的類別
        """
        self.obj_detect_mode = False # 預設模式為：虛擬圍籬
        # self.cam_areas = config.get_cam_areas(self.obj_detect_mode) # TODO 這邊之後要用 queue的訊號 來更改 process中的 maskArr的值
        self.least_safe_distance = 400 # TODO 尚未能從前端修改
        ## Wen 新增 模式切分出來 儲存圍籬區域不同========================#
        self.cam_vf_areas = config.get_cam_areas(False) 
        self.cam_obj_areas = config.get_cam_areas(True)
        # self.alarm_level_set = config.get_alarm_level_setting()
        # self.alarm_channel = config.get_alarm_channel()
        self.vf_alarm_set, self.obj_alarm_set = config.get_alarm_setting('vf'), config.get_alarm_setting('obj')
        self.current_model_weights, self.current_model_class = config.get_model_config(self.obj_detect_mode) 
        #================================================================#

#取得模型權重清單
def get_model_weights_list():
    weights_list = []
    file_list = os.listdir(MODEL_WEIGHTS_DIR)
    for f in file_list:
        if f.endswith(".pt"):
            f = f.replace('.pt','')
            if os.path.isfile(MODEL_WEIGHTS_DIR+'\\'+f+'.yml') or os.path.isfile(MODEL_WEIGHTS_DIR+'\\'+f+'.yaml'):
                weights_list.append(f)
    return weights_list

#取得模型類別清單
def get_model_class_list(weights):
    yml_file_path = os.path.join(MODEL_WEIGHTS_DIR,'%s.yml' % weights)
    yaml_file_path = os.path.join(MODEL_WEIGHTS_DIR,'%s.yaml' % weights)
    data = None
    try:
        with open(yml_file_path, 'r') as stream:
            data = yaml.safe_load(stream)
    except Exception as e:
        with open(yaml_file_path, 'r') as stream:
            data = yaml.safe_load(stream)

    if data is None:
        logger.error(f'測距/物件:取得模型類別的list失敗, {e}')
    
    class_list = data['names']
    return class_list

# 取得異常發報訊息log
def get_alarm_log_json():
    try:
        log_json = load_json(CONFIG_DIR, 'alarm_msg_log.json')
        return log_json
    except Exception as e:
        logger.error(f'測距/物件:取得異常發報訊息log失敗, {e}')
        return {}

# 回傳影像
def return_image_to_web(target_dir, filename):
    filepath = os.path.join(IMG_DIR, target_dir, f'{filename}.jpg')
    return process_response_image(filepath)

# 取得或儲存過往的圖片 (按下 影像切換(過往) 按鈕時)
def save_past_setting_area_img(obj_dect_mode, img):
    #obj_dect_mode 為 bool
    if not obj_dect_mode:
        PAST_AREA_IMG = VF_PAST_AREA_IMG
    else:
        PAST_AREA_IMG = OBJ_PAST_AREA_IMG
    cv2.imwrite(PAST_AREA_IMG, img)

def get_past_setting_area_img(obj_dect_mode):
    if not obj_dect_mode:
        PAST_AREA_IMG = VF_PAST_AREA_IMG
    else:
        PAST_AREA_IMG = OBJ_PAST_AREA_IMG
    img = cv2.imread(PAST_AREA_IMG)
    return img

def save_past_setting_bili_img(obj_dect_mode, img):
    if not obj_dect_mode:
        PAST_BILI_IMG = VF_PAST_BILI_IMG
    else:
        PAST_BILI_IMG = OBJ_PAST_BILI_IMG
    cv2.imwrite(PAST_BILI_IMG, img)    

def get_past_setting_bili_img(obj_dect_mode):
    if not obj_dect_mode:
        PAST_BILI_IMG = VF_PAST_BILI_IMG
    else:
        PAST_BILI_IMG = OBJ_PAST_BILI_IMG
    img = cv2.imread(PAST_BILI_IMG)
    return img


# alarm_play 會儲存的 照片 及 聲音
def save_alarm_img(img):
    cv2.imwrite(ALARM_IMG, img)


def get_history_mp3_json():
    if os.path.isfile(HISTORY_MP3_JSON_PATH):
        return load_json(AUDIO_DIR, HISTORY_MP3_JSON_FILENAME)
    else:
        # 初始化 json
        aDict = dict()
        aDict['history'] = list()
        return aDict

def save_history_mp3_json(aDict):
    save_json(aDict, AUDIO_DIR, HISTORY_MP3_JSON_FILENAME)


from ..global_object import StreamingQueues, GlobalVar
# def vf_detect_init(q:StreamingQueues, vfvar:VfVar, gvar:GlobalVar):
def vf_detect_init(q:StreamingQueues, vfvar:VfVar, gvar=None):   
    from ..yolov5.vf_yolov5_detect import Detector
    from vgpy.utils.logger import create_logger
    logger = create_logger()

    q.c_queue.put('vf_init') # 初始化中，透過這個讓攝影機的畫面在模型載入前，能先顯示原始畫面    
    # TODO 切換辨識權重 yolov5s 或 fab_person
    vf_detector = Detector(vfvar, gvar, conf_thres=0.8, weights=vfvar.current_model_weights) # "yolov5s.pt"
    # vf_detector = Detector(vfvar, conf_thres=0.6, weights="fab_person.pt") # demo 影片用的參數
    
    # q.c_queue.get() # 把前面 put的訊號釋放掉，代表模型已載入完成，攝影機會能開始將照片輸入進來做辨識了
    # q.c_queue.task_done()
    while True:
        if not q.c_queue.empty():
            signal = q.c_queue.get(block=False)
            if type(signal) is tuple:
                signal, new_vfvar = signal
                '''
                    NOTE
                    當變更圍籬區域，觸發SIGNAL，將 vfapi中的變數 cam_areas(String Types)
                    帶給 vf_detector.update_fences => 用來將座標的str 格式轉為 int
                    最後把 int格式的座標 更新至 vf_detector(自身)的變數 fences_int 
                '''
                if signal is VfVar.SIGNAL.CHANGE_VF_AREAS:
                    new_areas = new_vfvar.cam_vf_areas
                    vfvar.cam_areas = new_areas
                    vf_detector.update_fences(new_areas)
                    print("CHANGE_VF_AREAS true")
                elif signal is VfVar.SIGNAL.CHANGE_OBJ_AREAS:
                    new_areas = new_vfvar.cam_obj_areas
                    vfvar.cam_areas = new_areas
                    vf_detector.update_fences(new_areas)
                    print("CHANGE_OBJ_AREAS true")
                elif signal is VfVar.SIGNAL.CHANGE_VF_ALARM_SET:
                    new_set = new_vfvar.vf_alarm_set
                    vf_detector.update_alarm_setting(new_set)
                    print("CHANGE_VF_ALARM_SET true")
                elif signal is VfVar.SIGNAL.CHANGE_OBJ_ALARM_SET:
                    new_set = new_vfvar.obj_alarm_set
                    vf_detector.update_alarm_setting(new_set)
                    print("CHANGE_OBJ_ALARM_SET true")
                # elif signal is VfVar.SIGNAL.CHANGE_ALARM_CHANNEL:
                #     vfvar.alarm_channel = new_vfvar.alarm_channel
                elif signal is VfVar.SIGNAL.CHANGE_MODEL_WEIGHTS:
                    new_weights = new_vfvar.current_model_weights
                    new_classes = new_vfvar.current_model_class
                    vfvar.current_model_weights, vfvar.current_model_class = new_weights, new_classes 
                    vf_detector.reload_model_weights(new_weights,new_classes)
                    print("CHANGE_MODEL_WEIGHTS true")
                elif signal is VfVar.SIGNAL.CHANGE_DETECT_MODE:
                    vfvar.obj_detect_mode = new_vfvar.obj_detect_mode  
                    if vfvar.obj_detect_mode:            
                        new_areas = new_vfvar.cam_obj_areas
                        new_alarm_set = new_vfvar.obj_alarm_set
                        print("obj_detect_mode")  
                    else:
                        new_areas = new_vfvar.cam_vf_areas
                        new_alarm_set = new_vfvar.vf_alarm_set
                        print("vf_detect_mode")
                    new_weights, new_classes  = config.get_model_config(vfvar.obj_detect_mode)  
                    vf_detector.reload_model_weights(new_weights,new_classes)
                    vfvar.cam_areas = new_areas
                    vf_detector.update_fences(new_areas)
                    vf_detector.update_alarm_setting(new_alarm_set)
                    print("CHANGE_DETECT_MODE true")

            elif signal is VfVar.SIGNAL.ALARM_CLEAR:
                vf_detector.alarm_var.notify_disable= False
                print("ALARM_CLEAR true")
            elif signal is VfVar.SIGNAL.SETTING_PAGE_ON:
                vf_detector.alarm_flag = 0 # 開啟設定頁，就停止發報
                if not q.d_queue.empty(): # 若是剛好有送圖進來，就先把圖取出，避免下面又辨識的，又更新到 alarm flag
                    im0 = q.d_queue.get()
                    q.r_queue.put(im0)
                    print("not q.d_queue.empty")
                print("SETTING_PAGE_ON true")

            elif signal is GlobalVar.SIGNAL.END_PROCESS:
                # 流程結束
                print('GlobalVar.SIGNAL.END_PROCESS')
                break
            
            elif signal is GlobalVar.SIGNAL.CHANGE_CAM_ID:
                new_camid, new_camwh = q.c_queue.get(block=True, timeout=5)
                gvar.cam_id = new_camid
                gvar.cam_wh = new_camwh

            try:
                logger.info(f'測距/物件:收到[{signal.name}]的signal')
            except:
                pass
            
            q.c_queue.task_done()
            continue

            # detector.update_fences(maskArr) # 更新 新設定的圍籬範圍到 偵測的程式
        if not q.d_queue.empty():

            start = time.time()

            img = q.d_queue.get()

            result_img, im0 = vf_detector.detect_img(img)
            if result_img is None:
                result_img = im0
            
            q.r_queue.put(result_img)
            print(f"time: ", time.time()-start)
        # print('vf_detect_init cycle run')
    
    q.clear_and_done_control_queue()
    print('clear_and_done_control_queue()')