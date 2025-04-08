from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 使用較小的模型
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"

# 初始化模型和 tokenizer
def init_model():
    try:
        print("正在下載 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("正在下載模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return tokenizer, model
    except Exception as e:
        print(f"模型初始化失敗: {str(e)}")
        return None, None

# 生成回應的函數
def generate_response(input_text, tokenizer=None, model=None):
    try:
        if tokenizer is None or model is None:
            tokenizer, model = init_model()
            if tokenizer is None or model is None:
                return "模型初始化失敗，請稍後再試。"
        
        print(f"正在處理輸入: {input_text}")
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        print("正在生成回應...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=10000,
            temperature=0.7,
            do_sample=True
        )
        
        # 只顯示回應，不顯示輸入問題
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除輸入問題
        response = response.replace(input_text, "").strip()
        
        return response
    except Exception as e:
        print(f"生成回應時發生錯誤: {str(e)}")
        return f"生成回應時發生錯誤: {str(e)}"

# 測試用
if __name__ == "__main__":
    # 初始化模型
    tokenizer, model = init_model()
    
    # 測試輸入
    input_text = "解釋一下什麼是AI"
    response = generate_response(input_text, tokenizer, model)
    
    print("\n回應：")
    print("=====================================")
    print(response)
    print("=====================================")
