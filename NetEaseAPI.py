'''Get song urls by singers' name.
    Using Multi-Thread. '''
__author__ = 'GhostAnderson'

import requests
import queue

user_track_info_template = 'http://localhost:3000/user/playlist?uid={}'
user_follow_template = 'http://localhost:3000/user/follows?uid={}'
track_info_template = 'http://localhost:3000/playlist/track/all?id={}&limit={}&offset=1'

import collections
track_song_map = collections.defaultdict(list)

visited_user = set()
visited_track = set()
user_queue = queue.Queue()

start_point = 45240497
user_queue.put(start_point)

start_point = 114651616
user_queue.put(start_point)


import time

import json
try:
    while not user_queue.empty():
        print(f'Now queue size: {user_queue.qsize()}, {len(track_song_map)}/50000 tracks has gathered')
        if len(track_song_map) >= 50000:
            break

        user = user_queue.get()
        if user in visited_user: continue
        user_tracks_link = user_track_info_template.format(user)
        result = json.loads(requests.get(user_tracks_link).text)
        if result['code'] == 200:
            filtered_sound_list = filter(lambda x: x[1] >= 10 ,map(lambda x: (x['id'], x['trackCount']), result['playlist']))
            visited_user.add(user)
            for sound_track_id, length in filtered_sound_list:
                if sound_track_id in visited_track: continue
                track_info_link = track_info_template.format(sound_track_id, 1000)
                result = json.loads(requests.get(track_info_link).text)
                if result['code'] == 200:
                    songs = list(map(lambda x: (x['name'], x['id'], x['ar'][0]['name']), result['songs']))
                    track_song_map[sound_track_id] = songs
                    visited_track.add(sound_track_id)
            
            user_follow_link = user_follow_template.format(user)
            result = json.loads(requests.get(user_follow_link).text)
            if result['code'] == 200:
                user_ids = list(map(lambda x: x['userId'], result['follow']))
                list(map(user_queue.put, user_ids))
        else:
            break

        time.sleep(5)
except:
    pass
finally:
    import pickle
    pickle.dump(track_song_map, open('track_song_map.pkl', 'wb'))
    pickle.dump(visited_user, open('visited_user.pkl', 'wb'))
    pickle.dump(visited_track, open('visited_track.pkl', 'wb'))
    pickle.dump(list(user_queue.queue), open('user_queue.pkl', 'wb'))

# with open('visited_user.txt', 'w') as visitied_user_file, open('visited_track.txt', 'w') as visited_track_file, open('track_song_map.txt', 'w') as track_song_map_file:
#     list(map(lambda x:visitied_user_file.write(f'{x}\n'), visited_user))
#     list(map(lambda x:visited_track_file.write(f'{x}\n'), visited_track))

#     for sound_track, songs in track_song_map.items():
#         track_song_map_file.write('{}\t{}\n'.format(sound_track, ','.join(map(str, songs))))