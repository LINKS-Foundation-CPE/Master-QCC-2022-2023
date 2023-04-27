import requests as req
import configparser as readconf
import csv
import os


class readConfig():
    apikey=''
    apilink=''
    username=''
    apiexample=''
    time=''
    def __init__(self):
        basedir = os.path.dirname(__file__)
        settingFile=os.path.join(basedir, 'setting.ini')
        sec=readconf.ConfigParser()
        self.configs=sec.read(settingFile) 
        self.apikey=sec['DEFAULT']['APIKEY']    
        self.apilink=sec['DEFAULT']['APILINK']    
        self.username=sec['DEFAULT']['User']   
        self.apiexample=sec['DEFAULT']['ApiExample'] 
        self.csvfile=sec['DEFAULT']['CSVFILE'] 
        self.debugmode=sec['DEFAULT'].getboolean('Debug') 

    def readApiLink(self):
        print(self.apilink)
        return self.apilink

    def readApiKey(self):
        print(self.apilink)
        return self.apilink

    def readUserName(self):
        print(self.username)
        return self.username
        
    def readApiExample(self):
        print(self.apiexample)
        return self.apiexample


class apiPollution():
    apikey=''
    test=''
    apiexample=''
    time_now=''
    fullData={}
    fullData_now={}
    params={}
    header=[]
    data_dic={}
    def __init__(self):
        setting=readConfig()
        self.context=req.Session()
        self.basedir = os.path.dirname(__file__)
        self.csvfile = setting.csvfile
        self.apilink=setting.apilink
        self.apikey=setting.apikey
        self.apiexample=setting.apiexample
        self.debug=setting.debugmode
        self.params={"key":self.apikey}
        self.pm25_now=0
        self.pm10_now=0
        self.pm25=0
        self.pm10=0
        self.header=['datetime',"lat","lon","PM2.5","PM10"]
        if self.debug:
            print('apiPollution has been acceable')
        

    def apiTest(self):
        r = self.context.get(url=self.apiexample,params=self.params) 
        if r.status_code != 200:
            print(r.text)
            print("Please check your API key validity")
            return 1    
        else:
            return 0
    def addParams(self,key,value):
        self.params[key]=value
        if self.debug:
            print('New parameter added')  
        return self.params  
    # it returns pm2.5 and pm10 Hourly
    def queryPM(self,y,x):
        self.addParams(key="features",value="pollutants_concentrations")
        self.addParams(key="lat",value=y)
        self.addParams(key="lon",value=x)
        r = self.context.get(url=self.apilink+"/current-conditions",params=self.params)  
        self.fullData_now=r.json() 
        self.pm25_now=self.fullData_now['data']['pollutants']['pm25']['concentration']['value']
        self.pm10_now=self.fullData_now['data']['pollutants']['pm10']['concentration']['value']
        self.time_now=self.fullData_now['data']['datetime']
        self.lat_now=y
        self.lon_now=x
        if self.debug:
            print("the full object added to 'self.fullData_now'") 
        return self.pm25,self.pm10,self.time_now
    # only can come back to one mounth ago (not more)
    def queryHistoryPM(self,y,x,datetime):
        self.addParams(key="features",value="pollutants_concentrations")
        self.addParams(key="lat",value=y)
        self.addParams(key="lon",value=x)
        self.addParams(key="datetime",value=datetime)
        r = self.context.get(url=self.apilink+"/historical/hourly",params=self.params)  
        self.fullData=r.json() 
        self.pm25=self.fullData['data']['pollutants']['pm25']['concentration']['value']
        self.pm10=self.fullData['data']['pollutants']['pm10']['concentration']['value']
        self.time=datetime
        self.lat=y
        self.lon=x
        self.data_formated()
        if self.debug:
            print("the full object added to 'self.fullData'") 
        return self.pm25,self.pm10,self.time
    # generate dictionary for saving inside the csv
    def data_to_dic(self,key,value):
        self.data_dic[key]=value
        if self.debug:
            print(f'New parameter added to data_dic[{key}]={value}') 
    #self.header=['datetime',"lat","lon","pm2.5","pm10"]
    def data_formated(self):
        self.data_to_dic(key=self.header[0],value=self.time)
        self.data_to_dic(key=self.header[1],value=self.lat)
        self.data_to_dic(key=self.header[2],value=self.lon)
        self.data_to_dic(key=self.header[3],value=self.pm25)
        self.data_to_dic(key=self.header[4],value=self.pm10)
        if self.debug:
            print(f'data_formated ok!')
        return True
    # store historical query to csv
    
    def saveCSV(self):
        if os.path.isfile(os.path.join(self.basedir, self.csvfile)):
            with open(os.path.join(self.basedir, self.csvfile), mode='a') as csv_file:
                fieldnames = self.header
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(self.data_dic)
        else:
            with open(os.path.join(self.basedir, self.csvfile), mode='w') as csv_file:
                fieldnames = self.header
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(self.data_dic) 

        if self.debug:
            print(f'{self.data_dic} is stored!')
        return True    