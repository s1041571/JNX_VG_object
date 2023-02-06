import json
import os
import math
from flask import config
from ...utils.json_utils import load_json, save_json

current_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR = current_dir # config 資料夾

# DANGER_AREA_COORD_JSON = os.path.join(CONFIG_DIR, 'dangerAreaCoord.json')
# MASK_COORD_TXT = os.path.join(CONFIG_DIR, 'ROI', 'mask_coord.txt')
## 8/19 更新 ======================================================================#
VF_MASK_COORD_TXT = os.path.join(CONFIG_DIR, 'ROI', 'mask_vf_coord.txt')
OBJ_MASK_COORD_TXT = os.path.join(CONFIG_DIR, 'ROI', 'mask_obj_coord.txt')
VF_DANGER_AREA_COORD_JSON = os.path.join(CONFIG_DIR, 'dangerAreaCoord_VF.json')
OBJ_DANGER_AREA_COORD_JSON = os.path.join(CONFIG_DIR, 'dangerAreaCoord_OBJ.json')
VF_BILI_JSON = os.path.join(CONFIG_DIR, 'Bili_VF.json')
VF_BILI_COORD_JSON = os.path.join(CONFIG_DIR, 'Bili_Coord_VF.json')
OBJ_BILI_JSON = os.path.join(CONFIG_DIR, 'Bili_OBJ.json')
OBJ_BILI_COORD_JSON = os.path.join(CONFIG_DIR, 'Bili_Coord_OBJ.json')
ALARM_LEVEL_JSON = os.path.join(CONFIG_DIR, 'alarm_level_setting.json')
ALARM_CHANNEL_JSON = os.path.join(CONFIG_DIR, 'alarm_channel.json')
ALARM_SETTING_JSON = os.path.join(CONFIG_DIR, 'alarm_setting.json') #包含兩模式的距離等級&通報頻道
LAST_MODEL_CONFIG = os.path.join(CONFIG_DIR,'last_model_config.json')
#==============================================================================#
DATATABLE_JSON = os.path.join(CONFIG_DIR, 'dataTable.json')


def from_canvas_to_camera(canvas_areas_coords, canvas_wh, cam_wh):
    
    # 把網頁中 canvas設置的點座標，轉換成相機解析度的座標
    canvas_w, canvas_h = canvas_wh
    cam_w, cam_h = cam_wh
    
    cam_areas = []
    for points in canvas_areas_coords:
        tmp_area_points = []
        for point in points:
            x,y = point          
            x = math.floor(int(x)/(canvas_w/cam_w))
            y = math.floor(int(y)/(canvas_h/cam_h))
            tmp_area_points.append((x, y))
        cam_areas.append(tmp_area_points)

    return cam_areas

#取得最後一次的模型 & 類別配置
def get_model_config(obj_detect_mode):
    mode = "obj" if obj_detect_mode else "vf"
    with open(LAST_MODEL_CONFIG,'r') as file:
        config_dict = json.load(file)
    return config_dict[mode]['weights'],config_dict[mode]['classes']

def save_model_config(mode, new_weights, new_classes):
    config_dict = load_json(CONFIG_DIR, 'last_model_config')
    json_data ={
        "weights": new_weights, 
        "classes": new_classes
    }
    config_dict[mode] = json_data
    save_json(config_dict, CONFIG_DIR, 'last_model_config')

#取得ALARM相關設定值
def get_alarm_setting(mode):
    with open(ALARM_SETTING_JSON,'r') as file:
        r_json = json.load(file)
        set_json = r_json[mode]
    return set_json

#儲存ALARM相關設定值
def save_alarm_setting(mode, item, data):
    current_json = load_json(CONFIG_DIR, 'alarm_setting')
    if item =='level':
        w_json = {
            "lv1": eval(data[0]),
            "lv2": eval(data[1]),
            "lv3": eval(data[2]) 
        }
    else:   #item = channel, alarm_threhold, log
        w_json = data
    
    if current_json[mode].get(item):
        current_json[mode][item].update(w_json)
    else:
        current_json[mode][item] = w_json
        
    save_json(current_json, CONFIG_DIR, 'alarm_setting')
    return w_json

#取得警報距離設定值
def get_alarm_level_setting():
    with open(ALARM_LEVEL_JSON,'r') as file:
        level_dict = json.load(file)
    return level_dict

#儲存警報距離設定值
def save_alarm_level_setting(level_list):
    json_data = {
        "lv1": eval(level_list[0]),
        "lv2": eval(level_list[1]),
        "lv3": eval(level_list[2]) 
    }
    with open(ALARM_LEVEL_JSON, 'w') as file:
        json.dump(json_data, file)
    return json_data

def get_alarm_channel():
    with open(ALARM_CHANNEL_JSON, 'r') as file:
        alarm_channel_dict = json.load(file)
    return alarm_channel_dict

#儲存通報群組設定
def save_alarm_channel(json_data):
    with open(ALARM_CHANNEL_JSON, 'w') as file:
        json.dump(json_data, file)
    


# 取得圍籬座標
def get_cam_areas(obj_dect_mode):
    if obj_dect_mode:
        MASK_COORD_TXT = OBJ_MASK_COORD_TXT
    else:
        MASK_COORD_TXT = VF_MASK_COORD_TXT
    # 取得 攝影機 的解析度之下的圍籬的座標
    maskArr = []
    with open(MASK_COORD_TXT, 'r') as f:
        a = f.read()
        linesArr = a.split('/')
        for lines in linesArr:
            line = lines.split('\n')
            maskCoordArr=[]
            for l in line:
                if l !='':
                    l=l.replace(' ', '')
                    maskCoordArr.append(tuple(map(int, l.split(','))))
            if maskCoordArr:
                maskArr.append(maskCoordArr)
    # print(maskArr)
    return maskArr

def get_canvas_area_str(obj_dect_mode):
    if not obj_dect_mode:
        DANGER_AREA_COORD_JSON = VF_DANGER_AREA_COORD_JSON
    else:
        DANGER_AREA_COORD_JSON = OBJ_DANGER_AREA_COORD_JSON
    # 取得 網頁canvas 的解析度之下的圍籬的座標  (為字串，格式如下)
    # "{202,62 153,330 147,379 452,474 595,78 },{...},..."
    with open(DANGER_AREA_COORD_JSON, 'r', encoding='utf-8') as a:
        a_coord = json.load(a)
    
    areas_str = ''
    for points in a_coord['coord']:
        points_str = ''
        for point in points:
            points_str += ','.join(point) + ' '
        areas_str += '{%s},' % points_str    
    a_coord['coord'] = areas_str
    return a_coord

def save_fence_areas_ponits(obj_dect_mode, canvas_data, cam_areas):
    if not obj_dect_mode:
        DANGER_AREA_COORD_JSON = VF_DANGER_AREA_COORD_JSON
        MASK_COORD_TXT = VF_MASK_COORD_TXT
    else:
        DANGER_AREA_COORD_JSON = OBJ_DANGER_AREA_COORD_JSON
        MASK_COORD_TXT = OBJ_MASK_COORD_TXT
    # 儲存 canvas 解析度的圍籬座標
    # TODO 儲存 json時可以加縮排，會更好看
    with open(DANGER_AREA_COORD_JSON, 'w', encoding='utf-8') as f:
        data = {
            "coord": canvas_data[0],
            "img_width": int(canvas_data[1]),
            "img_height": int(canvas_data[2])
        }
        json.dump(data, f, ensure_ascii=False)

    # 儲存 攝影機 解析度的圍籬座標
    a = ''
    for points in cam_areas:
        tmp_area_points = []
        for point in points:
            x,y = point
            a += f"{x},{y}\n"
            tmp_area_points.append((x, y))
        a += "/"

    with open(MASK_COORD_TXT, 'w') as f:
        f.write(a)


def get_bili(obj_dect_mode = False):
    # 取比例尺
    if not obj_dect_mode:
        BILI_JSON = VF_BILI_JSON
    else:
        BILI_JSON = OBJ_BILI_JSON
    with open(BILI_JSON, 'r', encoding='utf-8') as b:
        bili = json.load(b)
    return bili

def get_bili_coord(obj_dect_mode):
    if not obj_dect_mode:
        BILI_COORD_JSON = VF_BILI_COORD_JSON
    else:
        BILI_COORD_JSON = OBJ_BILI_COORD_JSON
    with open(BILI_COORD_JSON, 'r', encoding='utf-8') as b:
        b_coord = json.load(b)
    return b_coord
    

def save_bili(obj_dect_mode, bili):
    if not obj_dect_mode:
        BILI_JSON = VF_BILI_JSON
    else:
        BILI_JSON = OBJ_BILI_JSON
    # with open(BILI_JSON, 'w') as f:
    #     json.dump(bili, f, ensure_ascii=False)   
    with open(BILI_JSON, 'w', encoding='utf-8') as f:
        data = {
            "bili":eval(bili['bili']),
            "coord": bili['coord'],
            "img_width": int(bili['img_width']),
            "img_height": int(bili['img_height'])
        }
        json.dump(data, f, ensure_ascii=False)        

        
def save_bili_coord(obj_dect_mode, bili_coord):
    if not obj_dect_mode:
        BILI_COORD_JSON = VF_BILI_COORD_JSON
    else:
        BILI_COORD_JSON = OBJ_BILI_COORD_JSON
    with open(BILI_COORD_JSON, 'w', encoding='utf-8') as f:
        json.dump(bili_coord, f, ensure_ascii=False)


def get_data_table():
    with open(DATATABLE_JSON, 'r', encoding='utf-8') as f:
        data_table = json.load(f)
    return data_table

def save_data_table(data_table_str):
    with open(DATATABLE_JSON, 'w', encoding='utf-8') as f:
        json.dump(data_table_str[1:-1], f, ensure_ascii=False)


