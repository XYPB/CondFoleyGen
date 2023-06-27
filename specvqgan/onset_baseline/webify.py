import os
import datetime
import sys
import shutil
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--imgsize', type=int, default=100)
    parser.add_argument('--num', type=int, default=10000)

    args = parser.parse_args()
    return args


# --------------------------------------  joint ----------------------------------- #
def create_audio_visual_sec(args, f, name):
    dir_list = [name for name in os.listdir(
        args.path) if os.path.isdir(os.path.join(args.path, name))]
    dir_list.sort()

    f.write('''<div align = "center">''')

    joint_sec = """
<h3>{}</h3>
<table>
<tbody>
<tr>
<th>Index #</th>
    """.format(name)
    for name in dir_list:
        joint_sec += '''\n<th>{}</th>'''.format(name)
    joint_sec += '''\n</tr>\n'''
    f.write(joint_sec)

    item_list = []
    count = []
    for i in range(len(dir_list)):
        file_list = os.listdir(os.path.join(args.path, dir_list[i]))
        file_list.sort()
        count.append(len(file_list))
        item_list.append(file_list)
    file_count = min(count)
    for j in range(min(file_count, args.num)):
        f.write('''<tr>\n''')
        for i in range(-1, len(dir_list)):
            if i == -1:
                f.write('''<td> sample #{} </td>'''.format(str(j)))
                f.write('\n')
            else:
                sample = os.path.join(dir_list[i], item_list[i][j])
                if sample.split('.')[-1] in ['wav', 'mp3']:
                    f.write('''    <td> <div align = "center"><audio controls=""><source src='{}' type="audio/{}"></audio></div> </td>'''.format(
                        sample, sample.split('.')[-1]))
                elif sample.split('.')[-1] in ['jpg', 'png', 'gif']:
                    f.write(
                        '''    <td> <div align = "center"><img src='{}' style="zoom:{}%" /></div> </td>'''.format(sample, args.imgsize))
                elif sample.split('.')[-1] in ['mp4', 'avi', 'webm']:
                    f.write('''    <td> <div align = "center"><video id='{}' controls height='400'><source src="{}" type="video/{}" preload = "none"></video> <p>Speed: <input type="text" size="5" value="1" oninput="document.getElementById('{}').playbackRate = parseFloat(event.target.value);"></p></div> </td>'''.format(
                        sample, sample, sample.split('.')[-1], sample))
                f.write('\n')
                # <video id='{}' controls><source src="{}" type="video/{}"></video>

        f.write('''</tr>\n''')

    f.write('''</tbody></table>\n''')
    f.write('''</div>\n''')


# --------------------------------------  Audio  ----------------------------------- #
def create_audio_sec(args, f, name):
    f.write('''<div align = "center">''')

    audio_sec = """
<h3>{}</h3>
<table>
<tbody>
<tr>
<th>Index #</th>
<th>Mixture</th> 
<th>Original audio #1</th>
<th>Original audio #2</th>
<th>Separated audio #1</th>
<th>Separated audio #2</th>
<th>regenerated audio mix</th>
<th>regenerated audio #1</th>
<th>regenerated audio #2</th>
</tr>\n
    """.format(name)
    f.write(audio_sec)
    folder_path = os.path.join(args.path, 'audio')
    dir_list = os.listdir(folder_path)
    dir_list.sort()
    audio_list = []
    for i in range(len(dir_list)):
        l = os.listdir(os.path.join(folder_path, dir_list[i]))
        l.sort()
        audio_list.append(l)

    for j in range(len(audio_list[0])):
        f.write('''<tr>\n''')
        for i in range(-1, len(dir_list)):
            if i == -1:
                f.write('''<td> audio #{} </td>'''.format(str(j)))
                f.write('\n')
            else:
                audio_path = os.path.join(
                    folder_path, dir_list[i], audio_list[i][j])
                f.write('''    <td> <audio controls=""><source src='{}' type="audio/{}"></audio> </td>'''.format(
                    audio_path, audio_path.split('.')[-1]))
                f.write('\n')
        f.write('''</tr>\n''')

    f.write('''</tbody></table>\n''')
    f.write('''</div>\n''')


# --------------------------------------  Image ----------------------------------- #
def create_image_sec(args, f, name):
    f.write('''<div align = "center">''')

    image_sec = """
<h3>{}</h3>
<table>
<tbody>
<tr>
<th>Index #</th>
<th>Mixture Spec </th> 
<th>Original Spec #1</th>
<th>Original Spec #2</th>
<th>Separated Spec #1</th>
<th>Separated Spec #2</th>
</tr>\n
    """.format(name)

    f.write(image_sec)
    folder_path = os.path.join(args.path, 'spec_img')
    dir_list = os.listdir(folder_path)
    dir_list.sort()
    image_list = []
    for i in range(len(dir_list)):
        l = os.listdir(os.path.join(folder_path, dir_list[i]))
        l.sort()
        image_list.append(l)

    for j in range(len(image_list[0])):
        f.write('''<tr>\n''')
        for i in range(-1, len(dir_list)):
            if i == -1:
                f.write('''<td> audio #{} </td>'''.format(str(j)))
                f.write('\n')
            else:
                img_path = os.path.join(
                    folder_path, dir_list[i], image_list[i][j])
                f.write('''    <td> <div align = "center"><img src='{}' style="zoom:{}%" /></div> </td>'''.format(
                    img_path, 175))
                f.write('\n')
        f.write('''</tr>\n''')

    f.write('''</tbody></table>\n''')
    f.write('''</div>\n''')

# --------------------------------------  Video ----------------------------------- #


def create_video_sec(args, f, name):
    f.write('''<div align = "center">''')

    video_sec = """
<h3>{}</h3>
<table>
<tbody>
<tr>
<th></th>
<th></th>
<th></th>
</tr>\n
    """.format(name)

    f.write(video_sec)
    # folder_path = os.path.join(args.path, 'videos')
    video_list = glob.glob('%s/*.mp4' % args.path)
    video_list.sort()

    columns = 3
    rows = len(video_list) // columns + 1

    for i in range(rows):
        f.write('''<tr>\n''')
        for j in range(columns):
            index = i * columns + j
            if index < len(video_list):
                video_path = video_list[i * columns + j]
                f.write('''    <td> <div align = "center"><h4>{}</h4><video width="480" onmouseover = "this.controls = true;" onmouseout = "this.controls = false;"><source src="{}" type="video/{}"></video></div> </td>'''.format(
                    video_path.split('/')[-1], video_path, video_path.split('.')[-1]))
                f.write('\n')

        f.write('''</tr>\n''')

    f.write('''</tbody></table>\n''')
    f.write('''</div>\n''')


def webify(args):
    html_file = os.path.join(args.path, 'index.html')
    f = open(html_file, 'wt')

    # head
    # <link rel="stylesheet" type="text/css" title="Cool stylesheet" href="style.css">
    head = """<!DOCTYPE html>
<html lang="en"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Listening and Looking - UM Owens Lab</title>
</head>
    """
    f.write(head)

    intro_sec = '''
<body data-gr-c-s-loaded="true">
<h1> Listening and Looking - UM Owens Lab </h1>
<h5> Creator: Ziyang Chen <br>
University of Michigan </h5>
<p> This page contains the results of experiment.</p>
'''
    f.write(intro_sec)
    # create_audio_sec(args, f, "Audio Separation")
    # create_image_sec(args, f, 'Spectorgram Visualization')
    # create_video_sec(args, f, 'CAM Visualization')
    create_audio_visual_sec(args, f, 'Stereo CRW')
    f.write('''</body>\n''')
    f.write('''</html>\n''')
    f.close()


if __name__ == "__main__":
    args = parse_args()
    webify(args)
    print('Webify Succeed!')
