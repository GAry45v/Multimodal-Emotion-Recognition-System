import librosa
import torch.utils.data as data
import torchaudio
from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
from gradio_client import Client
import cv2
import tempfile
import threading
import subprocess
from datetime import datetime
import sys
import io
from PIL import Image
import torchvision.transforms as transforms
from moviepy import VideoFileClip

from models.vaanet import VAANet
import torch
from torchvision import get_image_backend
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import os
import functools
from transforms.spatial import Preprocessing
from transforms.temporal import TSN
from openai import OpenAI
from torch.utils.data import DataLoader

# 重新配置标准输出为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)  # 允许所有来源，确保前端能够访问后端端点

model = VAANet(snippet_duration=16, sample_size=112, n_classes=8, seq_len=12, audio_embed_size=256,
                   audio_n_segments=16, pretrained_resnet101_path="D:/flaskProject/resnet-101-kinetics.pth", )
model = model.cuda()
log_dir = "save_30.pth"
checkpoint = torch.load(log_dir, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
# 初始化摄像头
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("[ERROR] 摄像头未成功初始化，请检查摄像头连接或驱动。")
    camera.release()

# 全局变量
recording = False
output_file = "output.mp4"
frame_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = int(camera.get(cv2.CAP_PROP_FPS)) if int(camera.get(cv2.CAP_PROP_FPS)) > 0 else 30
video_writer = None
API_KEY = "8b5380ffbc97c6ad0ab54f5635539b53823ef5fe"  # 从环境变量中获取API密钥
BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"  # 百度AI Studio的API地址

# 初始化 OpenAI 客户端（这里用的是OpenAI包，实际上需要使用百度API SDK）
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def generate_frames():
    """实时生成视频流帧"""
    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] 无法读取摄像头数据。")
            break
        else:
            # 如果正在录制，将帧写入视频文件
            if recording and video_writer:
                video_writer.write(frame)
            # 编码成 JPEG 格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def start_recording():
    """开始录制视频"""
    global recording, video_writer
    if not recording:
        recording = True
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        print("[INFO] 视频录制已开始。")

def stop_recording():
    """停止录制视频"""
    global recording, video_writer
    if recording:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        print(f"[INFO] 视频已保存为 {output_file}")

def preprocess_audio(audio_path):
    "Extract audio features from an audio file"
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    return mfccs

def process_video(video_path, audio_path, temporal_transform, spatial_transform,
                  preprocess_audio, loader, target_transform, video_id, need_audio=True):
    """
    处理单个视频文件，提取帧和音频数据。

    Args:
        video_path (str): 视频文件路径。
        audio_path (str): 音频文件路径。
        temporal_transform (callable): 时间维度变换函数。
        spatial_transform (callable): 空间维度变换函数。
        preprocess_audio (callable): 音频预处理函数。
        loader (callable): 帧加载函数。
        target_transform (callable): 标签变换函数。
        video_id (str): 视频标识符。
        need_audio (bool): 是否需要处理音频。

    Returns:
        tuple: (snippets, target, audios, visualization_item)
    """
    import cv2
    import torch
    import numpy as np
    # 从视频读取所有帧并生成索引
    video_frames = []
    capture = cv2.VideoCapture(video_path)
    index = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        frame = Image.fromarray(frame)  # 转换为PIL.Image对象
        video_frames.append((index, frame))  # 保存帧索引和帧数据
        index += 1
    capture.release()

    # 时间变换：对帧索引进行采样或调整
    frame_indices = [idx for idx, _ in video_frames]  # 提取帧索引
    snippets_frame_idx = temporal_transform(frame_indices)

    # 音频处理
    if need_audio:
        timeseries_length = 4096
        feature = preprocess_audio(audio_path).T  # 自定义的音频预处理
        k = timeseries_length // feature.shape[0] + 1
        feature = np.tile(feature, reps=(k, 1))  # 通过复制填充音频
        audios = feature[:timeseries_length, :]  # 截断到目标长度
        audios = torch.FloatTensor(audios)
    else:
        audios = []

    # 视频帧处理
    snippets = []
    for snippet_frame_idx in snippets_frame_idx:
        snippet = [video_frames[idx][1] for idx in snippet_frame_idx]  # 按索引提取帧数据
        snippets.append(snippet)

    # 随机化空间变换参数
    spatial_transform.randomize_parameters()
    snippets_transformed = []
    for snippet in snippets:
        snippet = [spatial_transform(img) for img in snippet]  # 应用空间变换
        snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)  # 转换维度
        snippets_transformed.append(snippet)

    # 合并片段
    snippets = torch.stack(snippets_transformed, 0)

    # 目标标签处理
    # target = target_transform(video_id)  # 假设标签与视频ID有关

    # 可视化信息
    visualization_item = [video_id]

    return snippets, audios, visualization_item

def predict_num(data):
    '''
    返回概率最高的情感
    '''
    num = data.argmax()
    emotion = ['生气', '期待', '厌恶', '恐惧', '快乐', '悲伤', '惊讶', '信任']
    suggestion = [
    "当你感到生气时，尝试深呼吸，冷静下来，避免做出冲动的决定。让自己暂时离开引发愤怒的情境，进行一些放松活动，帮助恢复冷静。与他人沟通时，保持理性和耐心，避免情绪化的言辞，以免伤害到别人和自己。",

    "当你感到期待时，保持积极心态，适度调整期望值，避免过度焦虑。享受过程，而不仅仅是专注于结果。时刻提醒自己，任何的努力和付出都会带来成长，无论结果如何，都值得庆祝和骄傲。",

    "面对厌恶情绪时，尝试理解产生这种情绪的原因，看看是否有办法改变环境或人际关系。适当远离让你感到不舒服的事物，减少负面情绪的影响。你也可以通过转移注意力，做一些喜欢的活动，帮助自己恢复平和。",

    "面对恐惧时，深呼吸，给自己时间去适应，逐步面对恐惧源。不要急于强迫自己去做，让自己逐步建立自信。寻求身边人的支持，或者寻求专业帮助，理解恐惧情绪的来源，帮助自己以更健康的方式应对。",

    "当你感到快乐时，享受当下的愉悦感受，保持积极心态，感激生活中的美好事物。与他人分享你的快乐，不仅能增进彼此的关系，也能增强自己的幸福感。把快乐当作动力，去鼓励自己和他人，创造更多正能量。",

    "当你感到悲伤时，允许自己感受这种情绪，不要压抑。适当寻求他人的支持，和亲密的人聊聊你的感受。做一些放松活动，如散步、听音乐、冥想等，帮助你逐渐恢复。记住，悲伤是人生的一部分，它会过去，随着时间的推移，你会变得更加坚强。",

    "当你感到惊讶时，接受这种情绪的存在，避免过于惊慌。冷静分析发生的事情，看看是否可以从中获得一些启示。适应变化，并采取适当的应对措施。改变往往带来新的机会，你可以将惊讶转化为对未知的好奇，去拥抱新的挑战。",

    "当你感到信任时，维持与他人的信任关系，建立更深的沟通和理解。信任是任何关系的基础，保持透明和真诚，可以让你们的关系更加稳固。不要轻易怀疑他人，给予他们机会和信任，同时也要谨慎地建立自己的边界。"
    ]
    result = emotion[num]
    result_suggestion = suggestion[num]
    return result, result_suggestion
    #for i in range(8):
    #    print("{0}:{1}".format(emotion[i],prob[0][i]))

def predict_prob(data):
    '''返回所有情感的概率，其加和为1'''
    prob =  F.softmax(data, dim=1)
    #print(prob[0])
    result = prob[0]
    return result

def save_to_history(emotion, suggestion, HISTORY_FILE):
    # 获取当前的日期和时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 打开历史记录文件（如果文件不存在，则创建）
    with open(HISTORY_FILE, 'a', encoding='utf-8') as file:
        file.write(f"日期：{current_time}\n")
        file.write(f"情绪：{emotion}\n")
        file.write(f"建议：{suggestion}\n\n")

@app.route('/get_history', methods=['GET'])
def get_history():
    HISTORY_FILE = r"D:\flaskProject\history_file.txt"
    if not os.path.exists(HISTORY_FILE):
        return jsonify({'error': '历史记录文件不存在'}), 404

    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as file:
            history_lines = file.readlines()

        history_data = []
        # 每3行数据组成一个记录（日期，情绪，建议）
        for i in range(0, len(history_lines), 4):  # 4行数据为一组
            date = history_lines[i].strip().split("：")[1]
            emotion = history_lines[i + 1].strip().split("：")[1]
            suggestion = history_lines[i + 2].strip().split("：")[1]
            history_data.append({
                'date': date,
                'emotion': emotion,
                'suggestion': suggestion
            })

        return jsonify(history_data)

    except Exception as e:
        return jsonify({'error': f'读取历史文件时出错: {str(e)}'}), 500

@app.route('/ask_gpt', methods=['POST'])
def get_gpt():
    try:
        # 从请求中获取 JSON 数据
        data = request.get_json()
        user_query = data.get('user_query')  # 提取用户问题

        if not user_query:
            return jsonify({"error": "问题不能为空"}), 400  # 如果没有问题，返回 400 错误

        # 向 ERNIE API 发送请求
        chat_completion = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': '你是一名情绪治疗师，请为提问者提供专业且温暖的情绪建议，安抚提问者的情绪。'},
                {'role': 'user', 'content': user_query}
            ],
            model="ernie-3.5-8k",  # 使用 ERNIE 3.5 模型
        )

        # 获取响应的内容
        answer = chat_completion.choices[0].message.content

        # 返回模型的响应
        return jsonify({'answer': answer})  # 返回 JSON 格式的答案

    except Exception as e:
        # 捕获所有异常并返回错误信息
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_collection', methods=['GET'])
def data_collection():
    return render_template('data_collection.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_record():
    start_recording()
    return jsonify({"message": "视频录制已开始"})

@app.route('/stop_recording', methods=['POST'])
def stop_record_endpoint():
    stop_recording()
    return jsonify({"message": f"视频已保存为 {output_file}"})

@app.route('/data_analysis', methods=['GET'])
def data_analysis():
    return render_template('data_analysis.html')

@app.route('/result_display', methods=['GET'])
def result_display():
    return render_template('result_display.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': '没有上传视频文件'}), 400

    video_file = request.files['video']

    # 保存视频文件到临时路径
    video_path = './tmp/uploaded_video.mp4'  # 临时存储路径
    video_file.save(video_path)
    print(f"Video file saved to {video_path}")

    video = VideoFileClip(video_path)
    # 提取音频
    audio = video.audio
    # 保存音频为 MP3 格式
    audio.write_audiofile("./tmp/audio/extracted_audio.mp3", codec='mp3')
    # 提取音频
    spatial_transform = Preprocessing(size=112, is_aug=False, center=False)
    temporal_transform = TSN(seq_len=12, snippet_duration=16, center=False)
    video_path = r'D:\flaskProject\tmp\uploaded_video.mp4'
    audio_path = r'D:\flaskProject\tmp\audio\extracted_audio.mp3'
    history_file = r"D:\flaskProject\history_file.txt"
    snippets, audios, visualization_item = process_video(video_path, audio_path, temporal_transform, spatial_transform,
                                                         preprocess_audio, loader=None, target_transform=None,
                                                         video_id=1, need_audio=True)
    snippets = snippets.unsqueeze(0).cuda()
    audios = audios.unsqueeze(0).cuda()
    outputs = model(snippets, audios)
    y_pred, alpha, beta, gamma = outputs
    result, result_suggestion = predict_num(y_pred)
    print(result)
    print(result_suggestion)
    save_to_history(result, result_suggestion, history_file)
    return jsonify({
        'emotion': result,
        'suggestion': result_suggestion
    })

if __name__ == '__main__':
    app.run(debug=True)
