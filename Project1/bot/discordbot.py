import discord
from discord.ext import commands
import os
from discord import FFmpegPCMAudio
import asyncio
import concurrent.futures
import yt_dlp
import re
import threading

# 設定 bot 指令前綴
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# 創建線程池
executor = concurrent.futures.ThreadPoolExecutor()

# 自動控制相關變數
AUTO_CONTROL = False
TARGET_CHANNEL_ID = 0  # 要加入的語音頻道ID
MUSIC_FILE = None  # 要播放的音樂檔案
BOT_TOKEN = None  # Discord Bot Token
bot_thread = None  # 用於存儲 bot 運行的線程

# YouTube下載選項
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

# bot 啟動事件
@bot.event
async def on_ready():
    print(f'{bot.user} 已上線!')
    if AUTO_CONTROL:
        await auto_control()

async def auto_control():
    """自動控制機器人行為"""
    try:
        # 獲取目標頻道
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        if not channel:
            print(f"找不到頻道 {TARGET_CHANNEL_ID}")
            return

        # 如果已經在語音頻道中，先離開
        for vc in bot.voice_clients:
            if vc.guild == channel.guild:
                await vc.disconnect()

        # 加入頻道
        voice_client = await channel.connect()
        print(f"已加入頻道 {channel.name}")

        # 播放音樂
        if MUSIC_FILE:
            try:
                # 檢查音樂文件是否存在
                music_path = os.path.join('music', MUSIC_FILE)
                if not os.path.exists(music_path):
                    print(f"找不到音樂文件: {music_path}")
                    return

                source = FFmpegPCMAudio(music_path)
                voice_client.play(source)
                print(f"開始播放 {MUSIC_FILE}")

                # 等待音樂播放完成
                while voice_client.is_playing():
                    await asyncio.sleep(1)

            except Exception as e:
                print(f"播放音樂時發生錯誤: {str(e)}")
            finally:
                # 離開頻道
                await voice_client.disconnect()
                print("已離開頻道")

    except Exception as e:
        print(f"自動控制發生錯誤: {str(e)}")

# 設定自動控制參數
def setup_auto_control(channel_id, music_file=None):
    """設定自動控制參數"""
    global AUTO_CONTROL, TARGET_CHANNEL_ID, MUSIC_FILE
    AUTO_CONTROL = True
    TARGET_CHANNEL_ID = channel_id
    MUSIC_FILE = music_file
    # 如果 bot 已經在運行，立即執行 auto_control
    if bot.is_ready():
        asyncio.run_coroutine_threadsafe(auto_control(), bot.loop)
    print(f"已設置自動控制 - 頻道: {channel_id}, 音樂: {music_file}")

# 加入語音頻道指令
@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f'已加入 {channel} 頻道')
    else:
        await ctx.send('你必須先加入一個語音頻道!')

# 離開語音頻道指令  
@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send('已離開語音頻道')
    else:
        await ctx.send('Bot 不在任何語音頻道中')

# 播放本地音樂或YouTube音樂指令
@bot.command()
async def play(ctx, url):
    if not ctx.voice_client:
        if ctx.author.voice:
            await ctx.author.voice.channel.connect()
        else:
            await ctx.send('你必須先加入一個語音頻道!')
            return
            
    try:
        # 檢查是否為YouTube網址
        if re.match(r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+', url):
            # 下載YouTube音樂
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                url2 = info['url']
                source = FFmpegPCMAudio(url2, **{
                    'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5',
                    'options': '-vn'
                })
                await ctx.send(f'正在播放YouTube音樂: {info["title"]}')
        else:
            # 播放本地音樂
            source = FFmpegPCMAudio(f'../music/{url}')
            await ctx.send(f'正在播放本地音樂: {url}')
            
        ctx.voice_client.play(source)
    except Exception as e:
        await ctx.send(f'播放失敗: {str(e)}')

# 暫停播放指令
@bot.command()
async def pause(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.pause()
        await ctx.send('音樂已暫停')
    else:
        await ctx.send('目前沒有正在播放的音樂')

# 繼續播放指令
@bot.command()
async def resume(ctx):
    if ctx.voice_client and ctx.voice_client.is_paused():
        ctx.voice_client.resume()
        await ctx.send('繼續播放音樂')
    else:
        await ctx.send('音樂未被暫停')

# 停止播放指令
@bot.command()
async def stop(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send('音樂已停止')
    else:
        await ctx.send('目前沒有正在播放的音樂')

def run_bot():
    """在新線程中運行 bot"""
    if not BOT_TOKEN:
        print("錯誤：未設置 Bot Token")
        return
    try:
        bot.run(BOT_TOKEN)
    except Exception as e:
        print(f"Bot 運行出錯: {str(e)}")

def start_bot(token):
    """啟動 Discord Bot"""
    global BOT_TOKEN, bot_thread
    BOT_TOKEN = token
    if bot_thread is None or not bot_thread.is_alive():
        bot_thread = threading.Thread(target=run_bot)
        bot_thread.daemon = True  # 設置為守護線程
        bot_thread.start()
        return True
    return False

def stop_bot():
    """停止 Discord Bot"""
    global bot_thread, AUTO_CONTROL
    AUTO_CONTROL = False  # 停止自動控制
    if bot_thread and bot_thread.is_alive():
        asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
        bot_thread.join(timeout=5)  # 等待線程結束
        bot_thread = None
        return True
    return False

async def send_message_to_channel(channel_id, message):
    """向指定頻道發送消息"""
    try:
        channel = bot.get_channel(channel_id)
        if channel:
            await channel.send(message)
            return True
        else:
            print(f"找不到頻道 {channel_id}")
            return False
    except Exception as e:
        print(f"發送消息時發生錯誤: {str(e)}")
        return False

# 執行 bot
if __name__ == "__main__":
    TOKEN = 'YOUR_BOT_TOKEN_HERE'  # 請替換成您的 Discord Bot Token
    start_bot(TOKEN)