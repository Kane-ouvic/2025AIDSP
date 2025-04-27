import torch
from net import Net

# 1. 建立模型（注意，這裡不做 forward 也不呼叫 setTarget）
model = Net(ngf=128)
model.eval()
model.cuda()

# 2. 載入權重
model_dict = torch.load('./models/21styles.model')
new_state_dict = {k: v for k, v in model_dict.items() if 'running_mean' not in k and 'running_var' not in k}
model.load_state_dict(new_state_dict, strict=False)

# 3. 建立假輸入
dummy_input = torch.randn(1, 3, 480, 640).cuda()

# 4. 匯出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    './models/style_transfer.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)

print("✅ ONNX 匯出成功！")
