import discord
from discord.ext import commands
import os
from discord import FFmpegPCMAudio
import asyncio
from talk import init_model, generate_response
import concurrent.futures
import yt_dlp
import re

# 設定 bot 指令前綴
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# 初始化 AI 模型
tokenizer, model = init_model()

# 創建線程池
executor = concurrent.futures.ThreadPoolExecutor()

# 自動控制相關變數
AUTO_CONTROL = False
TARGET_CHANNEL_ID = 0  # 要加入的語音頻道ID
MUSIC_FILE = "3.mp3"  # 要播放的音樂檔案

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

# 處理非指令消息
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
        
    # 如果消息不是以 ! 開頭，就當作是問問題
    if not message.content.startswith('!'):
        try:
            # 顯示正在處理的訊息
            processing_msg = await message.channel.send("正在思考中...")
            
            # 在線程池中執行 AI 回應生成
            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(
                    executor,
                    generate_response,
                    message.content,
                    tokenizer,
                    model
                )
            except Exception as e:
                await message.channel.send(f"生成回應時發生錯誤: {str(e)}")
                return
            
            # 如果回應太長，分割成多個訊息
            if len(response) > 2000:
                chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(response)
                
            # 刪除處理中的訊息
            await processing_msg.delete()
            
        except Exception as e:
            await message.channel.send(f"發生錯誤: {str(e)}")
    
    # 處理其他指令
    await bot.process_commands(message)

# AI 對話指令
@bot.command()
async def ask(ctx, *, question):
    try:
        # 顯示正在處理的訊息
        processing_msg = await ctx.send("正在思考中...")
        
        # 在線程池中執行 AI 回應生成
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                executor,
                generate_response,
                question,
                tokenizer,
                model
            )
        except Exception as e:
            await ctx.send(f"生成回應時發生錯誤: {str(e)}")
            await processing_msg.delete()
            return
            
        # 刪除處理中的訊息
        await processing_msg.delete()
        
        # 將完整回應一次發送
        await ctx.send(response)
        
    except Exception as e:
        await ctx.send(f"發生錯誤: {str(e)}")

async def auto_control():
    """自動控制機器人行為"""
    try:
        # 獲取目標頻道
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        if not channel:
            print(f"找不到頻道 {TARGET_CHANNEL_ID}")
            return

        # 加入頻道
        voice_client = await channel.connect()
        print(f"已加入頻道 {channel.name}")

        # 播放音樂
        if MUSIC_FILE:
            source = FFmpegPCMAudio(f'../music/{MUSIC_FILE}')
            voice_client.play(source)
            print(f"開始播放 {MUSIC_FILE}")

            # 等待音樂播放完成
            while voice_client.is_playing():
                await asyncio.sleep(1)

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

# 執行 bot
if __name__ == "__main__":
    # 設定自動控制參數
    # 替換成您的頻道ID和音樂檔案名稱
    # setup_auto_control(851380042117283860, "3.mp3")  # 替換成實際的頻道ID和音樂檔案
    
    # 啟動機器人
    bot.run('')