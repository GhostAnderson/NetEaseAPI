import streamlit as st
import pickle
from inference import SASRecRecommender

import requests
import json

import random

# 载入映射
item2id = pickle.load(open('./item2id.pkl', 'rb'))
song2info = pickle.load(open('./song_id_info_map.pkl', 'rb'))

recommender = SASRecRecommender(model_path="./SASRec.pth", item2id=item2id)
user_track_info_template = 'http://localhost:3000/user/playlist?uid={}'
track_info_template = 'http://localhost:3000/playlist/track/all?id={}&limit={}&offset=1'
song_info_template = 'https://music.163.com/#/song?id={}'

import re

uid_re = re.compile(r'id=([0-9]+)')

st.set_page_config(page_title="🧠 SASRec 推荐系统", page_icon="🤖")
st.title("🧠 SASRec 匿名 Session 推荐系统")
st.markdown("请输入最近点击过的 item id，用英文逗号隔开（如：`item1,item8,item12`）")

user_input = st.text_input("👇 输入你的网易云音乐uid")

import time

def recommend(user_input):
    with open('saved.user.id.txt', 'a') as f:
        f.write(f'{user_input}\n')
    uid = uid_re.search(user_input).group(1)
    user_tracks_link = user_track_info_template.format(uid)
    result = json.loads(requests.get(user_tracks_link).text)

    st.success(f"你的 uid: {uid}")

    if result['code'] == 200:
        filtered_sound_list = list(filter(lambda x: x[1] >= 10 ,map(lambda x: (x['id'], x['trackCount']), result['playlist'])))

        sound_track_id = filtered_sound_list[0][0]
        track_info_link = track_info_template.format(sound_track_id, 1000)
        result = json.loads(requests.get(track_info_link).text)
        if result['code'] == 200:
            songs = list(map(lambda x: x['id'], result['songs']))
        print(list(map(lambda x: song2info[x], list(reversed(songs))[-100:])))
    
    random.seed(time.time())
    result = recommender.recommend(list(reversed(songs))[-100:], top_k=100)
    random.shuffle(result)
    return result[:20]

if st.button("🚀 推荐一下！"):
    session = [x.strip() for x in user_input.split(",") if x.strip()]
    if not session:
        st.warning("输入不能为空或格式错误！")
    else:
        topk = recommend(user_input)
        st.markdown("🎯 **推荐结果 Top 10：**")
        for idx, item in enumerate(topk, 1):
            st.markdown(f"{idx}. [**{song2info[int(item)][0]}** {song2info[int(item)][1]}]({song_info_template.format(item)})")