import cv2
from ..global_object import StreamingQueues, GlobalVar
from . import vf_backend as vfbackend
from vgpy.utils.logger import create_logger
logger = create_logger()

class WebController():
    def __init__(self):
        self.obj_detect_mode = False #物件辨識模式
        self.setFlag = 0
        self.shot = 0
        self.history = 0
        self.setTab = 0 # 1是虛擬圍籬設定中，2是比例尺設定中
        self.img_video_feed = None # 目前網頁上看到的畫面
        self.q:StreamingQueues = None
        
    def get_img(self, img0):
        if self.setFlag == 1 and self.shot == 0 :
            self.img_video_feed = img0

        elif self.setFlag == 1 and self.shot == 1:
            self.img_video_feed = img0

        elif self.setFlag == 1 and self.history == 1 and self.shot == 2 :
            if int(self.setTab) == 1:
                self.img_video_feed = vfbackend.get_past_setting_area_img(self.obj_detect_mode)
            elif int(self.setTab) == 2:
                self.img_video_feed = vfbackend.get_past_setting_bili_img(self.obj_detect_mode)
        
        elif self.setFlag == 0:
            self.q.d_queue.put(img0)
            try:
                results = self.q.r_queue.get()
                self.img_video_feed = results
            except Exception as e:
                logger.error(f'測距/物件:cannot get result_queue, {e}')
                self.img_video_feed = None
        
        return self.img_video_feed
            


