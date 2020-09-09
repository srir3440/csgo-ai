#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as matplt
import sys
import pygame
from pygame.locals import *
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


start=0
end=1

#for training use the following line

dataframes=list(pd.read_json(f) for f in ['datasets/dataset_{:02}.json'.format(i) for i in range(start,end)])

#for testing use the following line

test_dataframes=[pd.read_json(sys.argv[1])]


# In[ ]:


def get_dataframe(dataframes):
    input_data=pd.concat(dataframes,ignore_index=True)
    input_data=input_data.loc[input_data.alive_players.str.len()!=0,:]
    input_data.reset_index(drop=True,inplace=True)
    return input_data

input_data=get_dataframe(dataframes)
test_data=get_dataframe(test_dataframes)


# In[ ]:


def display_text(msg, pos):
    global screen
    pygame.font.init()
    font = pygame.font.Font('freesansbold.ttf', 12)
    text = font.render(msg, True, (255, 255, 255))
    screen.blit(text, pos)

game_map_coords = {
    "de_cache":    (-2000, 3250),
    "de_dust2":    (-2476, 3239),
    "de_inferno":  (-2087, 3870),
    "de_mirage":   (-3230, 1713),
    "de_nuke":     (-3453, 2887),
    "de_overpass": (-4831, 1781),
    "de_train":    (-2477, 2392),
    "de_vertigo":  (-3168, 1762),
}
game_map_scales = {
    "de_cache":    5.5,
    "de_dust2":    4.4,
    "de_inferno":  4.9,
    "de_mirage":   5.0,
    "de_nuke":     7.0,
    "de_overpass": 5.2,
    "de_train":    4.7,
    "de_vertigo":  4.0,
}
screen=None
round_snapshot=None
dataset=None
map_name = None
game_map = None
game_map_lower = None
game_map_scale = None
game_map_coord = None


def coord(x, y):
    global game_map_coord,game_map_scale
    '''Calculate radar coordinates from game coordinates for the chosen map'''
    res = (int(round((x-game_map_coord[0]) / game_map_scale)),
           int(abs(round((y-game_map_coord[1]) / game_map_scale))))
    return res


def load_snapshot(index):
    '''Load round snapshot `index` from the dataset and select the map it was played on'''
    global round_snapshot,dataset, map_name, game_map, game_map_lower, game_map_scale, game_map_coord
    round_snapshot = dataset.loc[index]
    map_name = round_snapshot['map']
    game_map = pygame.image.load(
        'resources/overview/{}_radar.png'.format(map_name))
    game_map_lower = pygame.image.load('resources/overview/{}_lower_radar.png'.format(
        map_name)) if map_name in ["de_nuke"] else game_map
    game_map_scale = game_map_scales[map_name]
    game_map_coord = game_map_coords[map_name]



def render_player(player):
    global screen
    # Colors
    ct_color = (0, 0, 255)
    t_color = (255, 200, 0)
    
    if len(player['position_history'])!=0:
        pos = player['position_history'][-1]
        color = ct_color if player['team'] == 'CT' else t_color
        pygame.draw.circle(
            screen, color, coord(pos['x'], pos['y']), 7
            )

def render_smoke(smoke):
    global screen
    pos = smoke['position']
    pygame.draw.circle(
        screen, (127, 127, 127), coord(pos['x'], pos['y']), 20
    )

def render_molotov(molotov):
    global screen
    pos = molotov['position']
    pygame.draw.circle(
        screen, (255, 69, 0), coord(pos['x'], pos['y']), 20
    )

def render_bomb(pos):
    global screen
    pygame.draw.rect(
        screen, (255, 0, 0), pygame.Rect(coord(pos['x'], pos['y']), (5, 8))
    )


def render_frame():
    global round_snapshot
    for molotov in round_snapshot['active_molotovs']:
        render_molotov(molotov)
    for smoke in round_snapshot['active_smokes']:
        render_smoke(smoke)
    for player in round_snapshot['alive_players']:
        render_player(player)
    if round_snapshot['planted_bomb']:
        render_bomb(round_snapshot['planted_bomb']['position'])


# In[ ]:


def generate_images(dataframes,is_train):
    global game_map,game_map_lower,screen,round_snapshot,dataset
    total_snaps=0
    for ds in dataframes:
        round_snapshot_index = 0
        dataset=ds
        round_snapshot = ds.loc[round_snapshot_index]
        num_snapshots = len(ds)
        load_snapshot(round_snapshot_index)

        # Init pygame
        pygame.init()

        # Screen
        screen_width = 1028
        screen_height = 1028
        screen = pygame.display.set_mode([screen_width, screen_height])
        clock = pygame.time.Clock()

        tooltip = ''
        for i in range(0,num_snapshots):
            round_snapshot_index = i
            load_snapshot(round_snapshot_index)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_l]:
                game_map, game_map_lower = game_map_lower, game_map
            if keys[pygame.K_ESCAPE]:
                pygame.quit()
                sys.exit()

            screen.fill((0, 0, 0))
            screen.blit(game_map, (0, 0))
            render_frame()
            if tooltip:
                display_text(tooltip, (800, 950))
                tooltip = ''

            pygame.display.update()
        
            #for training use the following line
            if is_train:
                imgname="round_snapshots/training/"+"dataset_"+str(total_snaps+round_snapshot_index)+".jpg"
        
            #for testing use the following line
            else:
                imgname="round_snapshots/test/"+"dataset_"+str(total_snaps+round_snapshot_index)+".jpg"
        
            pygame.image.save(screen,imgname)
            clock.tick(16)
    total_snaps+=num_snapshots


# In[ ]:


def get_series_from_dataset(ds,side,value=''):
    attrPlayers=map(lambda players:filter(lambda player:player['team']==side,players),ds['alive_players'])  
    if value!='':
        attrPlayers=map(lambda players:map(lambda player:player[value],players),attrPlayers)
    return list(map(list,attrPlayers))


# In[ ]:


def get_item_types_from_inventory(ds,side):
    values=map(lambda players:map(lambda objects:map(lambda object:object['item_type'],objects),players),get_series_from_dataset(ds,side,'inventory'))
    values=map(lambda players:map(lambda player:' '.join(player),players),values)
    return list(map(lambda p:' '.join(p),values))


# In[ ]:


def get_features_dataframe(ds):
    ds['dif_health']=(pd.Series(map(lambda p:sum(p),get_series_from_dataset(ds,'Terrorist','health')))-pd.Series(map(lambda p:sum(p),get_series_from_dataset(ds,'CT','health'))))
    ds['dif_armor']=(pd.Series(map(lambda p:sum(p),get_series_from_dataset(ds,'Terrorist','armor')))-pd.Series(map(lambda p:sum(p),get_series_from_dataset(ds,'CT','armor'))))
    no_ct_players=pd.Series(map(lambda p:len(p),get_series_from_dataset(ds,'CT')))
    ds['dif_no_of_players']=(pd.Series(map(lambda p:len(p),get_series_from_dataset(ds,'Terrorist')))-no_ct_players)
    ds['money_ct']=(pd.Series(map(lambda p:sum(p),get_series_from_dataset(ds,'CT','money')))/no_ct_players)
    ds['inventory_ct']=get_item_types_from_inventory(ds,'CT')
    ds['inventory_t']=get_item_types_from_inventory(ds,'Terrorist')
    ds['has_kit']=pd.Series(map(lambda p:any(p),get_series_from_dataset(ds,'CT','has_defuser')))
    ds.round_status_time_left[ds.round_status=='FreezeTime']/=20
    ds.round_status_time_left[ds.round_status=='Normal']/=115
    ds.round_status_time_left[ds.round_status=='BombPlanted']/=40
    return ds[['dif_health','dif_armor','dif_no_of_players','inventory_t','inventory_ct','round_status_time_left','round_status','has_kit','money_ct','map']]


# In[ ]:


from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
s_words=['c4','glock','p2000','usps']
cntVec=CountVectorizer(stop_words=s_words,max_features=30)
sc=StandardScaler()
le=LabelEncoder()
colTrans=ColumnTransformer(transformers=([('scaling',sc,make_column_selector(pattern='dif|time|money')),('ct_cnt_vectorizer',cntVec,'inventory_ct'),('t_cnt_vectorizer',cntVec,'inventory_t'),('round_status_encoder',OneHotEncoder(),['round_status']),('map_encoder',OneHotEncoder(),['map'])]),remainder='passthrough')


# In[ ]:


#map radar images for training data
generate_images(dataframes,True)

#map radar images for test data
generate_images(test_dataframes,False)


# In[ ]:


#for training use the following lines

X=get_features_dataframe(input_data)
y=le.fit_transform(input_data.round_winner)

#for testing use the following line

X_test=get_features_dataframe(test_data)


# In[ ]:


batch=64
def get_radar_image_array(length,is_train):
    
    #for training use the following line
    if is_train:
        img_name='round_snapshots/training/dataset_{}.jpg'

    #for testing use the following line
    else:
        img_name='round_snapshots/test/dataset_{}.jpg'

    image_data=[]
    for i in range(0,length):
        image=tf.keras.preprocessing.image.load_img(path=img_name.format(i),target_size=(64,64))
        image_data.append(tf.keras.preprocessing.image.img_to_array(image))
    image_data=np.array(image_data)
    return image_data


# In[ ]:


train_imggen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2)
test_imggen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#for training use the following line

num_input_data=colTrans.fit_transform(X).toarray()
image_input_data=get_radar_image_array(input_data.shape[0],True)

#for testing use the following line

num_test_data=colTrans.transform(X_test).toarray()
image_test_data=get_radar_image_array(test_data.shape[0],False)


# In[ ]:


def combined_generator(X1, X2, y,gen):
    gen1=gen.flow(X1,y,  batch_size=batch,seed=33,shuffle=False)
    gen2=gen.flow(X1,X2, batch_size=batch,seed=33,shuffle=False)
    while True:
        X1=gen1.next()
        X2=gen2.next()
        yield [X1[0], X2[1]], X1[1]
        
#for training use the following lines

from sklearn.model_selection import train_test_split
num_input_data_train,num_input_data_test,image_input_data_train,image_input_data_test,y_train,y_test=train_test_split(num_input_data,image_input_data,y,random_state=33,train_size=0.8)            
input_gen=combined_generator(image_input_data_train,num_input_data_train,y_train,train_imggen)
validation_gen=combined_generator(image_input_data_test,num_input_data_test,y_test,test_imggen)

#for testing use the following lines  

y_prr=np.zeros(num_test_data.shape[0])
test_gen=combined_generator(image_test_data,num_test_data,y_prr,test_imggen)


# In[ ]:


#Model architecture

inputs=tf.keras.Input(shape=(64,64,3),name='input_img')
num_data_input=tf.keras.Input(shape=(num_input_data_train.shape[1],),name='num_input')
x=tf.keras.layers.Conv2D(32,3,activation='relu',name='Convolution1',padding='same')(inputs)
x=tf.keras.layers.Conv2D(32,3,activation='relu',name='Convolution2',padding='same')(x)
x=tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,name='Pooling1')(x)
x=tf.keras.layers.Conv2D(64,3,activation='relu',name='Convolution3',padding='same')(x)
x=tf.keras.layers.Conv2D(64,3,activation='relu',name='Convolution4',padding='same')(x)
x=tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same',name='Pooling2')(x)
x=tf.keras.layers.Conv2D(128,3,activation='relu',name='Convolution5',padding='same')(x)
x=tf.keras.layers.Conv2D(128,3,activation='relu',name='Convolution6',padding='same')(x)
x=tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same',name='Pooling3')(x)
img_output=tf.keras.layers.Flatten()(x)
x=tf.keras.layers.Concatenate()([img_output,num_data_input])
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(1024,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
outputs=tf.keras.layers.Dense(1,activation='sigmoid')(x)
conv_model=tf.keras.Model([inputs,num_data_input],outputs,name='img_feature_gen')
conv_model.summary()
#tf.keras.utils.plot_model(conv_model,'conv_model1.jpg')


# In[ ]:


#compiling and fitting the model

conv_model.compile(loss='binary_crossentropy',metrics=['AUC','accuracy'],optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model_param=conv_model.fit(input_gen,epochs=75,validation_data=validation_gen,validation_steps=math.ceil(len(num_input_data_test)/batch),steps_per_epoch=math.ceil(len(num_input_data_train)/batch))


# In[ ]:


targets=conv_model.predict(test_gen,steps=math.ceil(len(num_test_data)/batch))
targets=pd.Series((targets.flatten())>0.5,dtype=int)
targets.to_json(sys.argv[2],orient='records')

