
import base64
from openai import AzureOpenAI

import io

api = {"key":
"f7bd69fb1e124bb79560a0e726fb4631",
'end':'https://hkust.azure-api.net/openai/deployments/{deployment-id}/chat/completions?api-version=2024-10-21',}

class AI:
    def __init__(self):
        self.client = AzureOpenAI(
        azure_endpoint = "https://hkust.azure-api.net",
        api_version = "2023-05-15",
        api_key = api['key'] #put your api key here
        )

    # 读取并编码图像
    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # 图像理解函数
    def analyze_image(self, images, prompt="请描述这些图片的内容。"):
        try:
            # 确保 images 是列表，即使传入单个图像
            if not isinstance(images, list):
                images = [images]
            
            # 准备多张图片的内容
            content = [{"type": "text", "text": prompt}]
            for image in images:
                # 确定图像格式，默认为 JPEG
                image_format = image.format if image.format in ["JPEG", "PNG"] else "JPEG"
                
                # 将 PIL Image 对象转换为 base64 编码
                buffered = io.BytesIO()
                image.save(buffered, format=image_format)
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # 设置 MIME 类型以匹配图像格式
                mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
                
                # 添加图像到内容
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # 使用支持图像的模型
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def get_response(self,message, instruction):
        response = self.client.chat.completions.create(
            model = 'gpt-4o',
            temperature = 1,
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": message}
            ]
        )
        
        # print token usage
        print(response.usage)
        # return the response
        return response.choices[0].message.content
    # 测试

    #image_path = "images/starteam-logo.png"  # 替换为你的图像路径
    #result = analyze_image(image_path)
    #print(result)

