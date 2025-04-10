# 呼叫 API 的範例程式碼
import requests

def call_llm_api(prompt):
    """
    呼叫 LLM API 的函式
    
    參數:
        prompt: 要發送給 LLM 的提示文字
    回傳:
        成功時回傳 LLM 的回應文字
        失敗時回傳錯誤訊息
    """
    # API 端點
    url = "http://web.nightcover.com.tw:60000/llm"
    
    # 準備請求資料
    data = {
        "prompt": prompt
    }
    
    try:
        # 發送 POST 請求
        response = requests.post(url, json=data)
        
        # 檢查回應狀態
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            return f"請求失敗: HTTP 狀態碼 {response.status_code}"
            
    except Exception as e:
        return f"發生錯誤: {str(e)}"

# 使用範例
if __name__ == "__main__":
    prompt = "根據悲傷這個情緒，寫一首詩，20字以內。"
    result = call_llm_api(prompt)
    print("LLM 回應:", result)



