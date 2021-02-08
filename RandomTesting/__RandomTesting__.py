import numpy as np
import pandas as pd
import gc
from tensorflow.keras import backend as K

def MSE(y, y_pred):
    mse = np.mean((y - y_pred)**2)
    return np.round(mse, 4)

def MAE(y, y_pred):
    mae = np.mean(np.abs(y - y_pred))
    return np.round(mae, 4)

def MAPE(y, y_pred): 
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    return np.round(mape, 4)

def Random_Testing(extract_nb, random_time, station_coordinate, EPA_test, extract5_list, extract10_list, extract15_list, extract20_list):
    n = 73
    if extract_nb == 0 :without_station_list = []
    if extract_nb == 1 :without_station_list = extract5_list
    if extract_nb == 2 :without_station_list = extract10_list        
    if extract_nb == 3 :without_station_list = extract15_list        
    if extract_nb == 4 :without_station_list = extract20_list        
    
    
    if extract_nb > 0:
        station_list = list(station_coordinate['SiteEngName'])
        for _ in range(len(without_station_list[random_time])):
            station_list.remove(without_station_list[random_time][_])

        station_coordinate_extract = station_coordinate[~station_coordinate['SiteEngName'].isin(station_list)]
        station_coordinate_exist = station_coordinate[station_coordinate['SiteEngName'].isin(station_list)]

        lats_extract = np.array(station_coordinate_extract['coordinate_lat'])
        lons_extract = np.array(station_coordinate_extract['coordinate_lon'])

        lats_exist = np.array(station_coordinate_exist['coordinate_lat']) 
        lons_exist = np.array(station_coordinate_exist['coordinate_lon'])

        matrix_test = np.zeros((EPA_test.shape[0],348,204,1))
        for k in range(n-len(without_station_list[random_time])):
            matrix_test[:,lats_exist[k],lons_exist[k],:] = EPA_test[:,lats_exist[k],lons_exist[k],:]
        gc.collect()
        K.clear_session()
        
    else:
        station_list = list(station_coordinate['SiteEngName'])
        station_coordinate_extract = station_coordinate[~station_coordinate['SiteEngName'].isin(station_list)]
        station_coordinate_exist = station_coordinate[station_coordinate['SiteEngName'].isin(station_list)]

        lats_extract = np.array(station_coordinate_extract['coordinate_lat'])
        lons_extract = np.array(station_coordinate_extract['coordinate_lon'])

        lats_exist = np.array(station_coordinate_exist['coordinate_lat']) 
        lons_exist = np.array(station_coordinate_exist['coordinate_lon'])

        matrix_test = np.zeros((EPA_test.shape[0],348,204,1))
        for k in range(n-len(without_station_list)):
            matrix_test[:,lats_exist[k],lons_exist[k],:] = EPA_test[:,lats_exist[k],lons_exist[k],:]
        gc.collect()
        K.clear_session()        

    return (matrix_test, lats_extract, lons_extract)


def RandomArea_Testing(extract_name, station_coordinate, EPA_test):
    def get_extract_area(extract_name):
        Taipei_area=['Dayuan','Zhongli','Pingzhen','Taoyuan','Longtan','Guanyin','Sanchong','Tucheng',
                     'Yonghe','Xizhi','Banqiao','Linkou','Tamsui','Cailiao','Xindian','Xinzhuang','Wanli',
                     'Shilin','Zhongshan','Guting','Songshan','Yangming','Wanhua','Keelung']

        Zhumiao_area=['Hsinchu','Zhudong','Hukou','Sanyi','Miaoli','Toufen']

        Taichung_area=['Dali','Xitun','Shalu','Zhongming','Fengyuan','Zhushan','Nantou','Puli','Changhua',
                       'Erlin','Xianxi']

        YunChiNan_area = ['Daliao','Lunbei','Mailiao','Taixi','Chiayi','Puzi','Xingang','Annan','Shanhua',
                          'Tainan','Xinying']

        KaoPing_area=['Douliu','Xiaogang','Renwu','Zuoying','Qianjin','Linyuan','Qianzhen','Meinong','Fuxing',
                      'Nanzi','Fengshan','Qiaotou','Pingtung','Hengchun','Chaozhou']


        if extract_name == 'Taipei_area': extract_area = Taipei_area
        elif extract_name == 'Zhumiao_area': extract_area = Zhumiao_area
        elif extract_name == 'Taichung_area': extract_area = Taichung_area
        elif extract_name == 'YunChiNan_area': extract_area = YunChiNan_area
        elif extract_name == 'KaoPing_area': extract_area = KaoPing_area
        else: 
            extract_area = []
            print('Not find area name')
        return(extract_area)    
    
    
    n = 73
    without_station_list = get_extract_area(extract_name)
        
    station_list = list(station_coordinate['SiteEngName'])
    for _ in range(len(without_station_list)):
        station_list.remove(without_station_list[_])
        
    station_coordinate_extract = station_coordinate[~station_coordinate['SiteEngName'].isin(station_list)]
    station_coordinate_exist = station_coordinate[station_coordinate['SiteEngName'].isin(station_list)]
    
    lats_extract = np.array(station_coordinate_extract['coordinate_lat'])
    lons_extract = np.array(station_coordinate_extract['coordinate_lon'])
    
    lats_exist = np.array(station_coordinate_exist['coordinate_lat']) 
    lons_exist = np.array(station_coordinate_exist['coordinate_lon'])

    matrix_test = np.zeros(EPA_test.shape)
    for k in range(n-len(without_station_list)):
        matrix_test[:,lats_exist[k],lons_exist[k],:] = EPA_test[:,lats_exist[k],lons_exist[k],:]
    gc.collect()
    return (matrix_test, lats_extract, lons_extract)


def calculate_extract_loss(epa_test, pred_test, coordinate_lat, coordinate_lon):
    mse_total = 0
    mae_total = 0
    mape_total  = 0
    for _ in range(len(coordinate_lat)):
        mse_total += MSE(pred_test[:,coordinate_lat[_],coordinate_lon[_]],
                         epa_test[:,coordinate_lat[_],coordinate_lon[_]])
        mae_total += MAE(pred_test[:,coordinate_lat[_],coordinate_lon[_]],
                         epa_test[:,coordinate_lat[_],coordinate_lon[_]])
        mape_total += MAPE(pred_test[:,coordinate_lat[_],coordinate_lon[_]],
                           epa_test[:,coordinate_lat[_],coordinate_lon[_]])
    average_mse = mse_total/len(coordinate_lat)
    average_mae = mae_total/len(coordinate_lat)
    average_mape = mape_total/len(coordinate_lat)
    return(average_mse, average_mae, average_mape)