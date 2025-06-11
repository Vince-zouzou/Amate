import subprocess
import sys
import os

def check_dependencies():
    """检查依赖是否已安装"""
    try:
        import streamlit
        import openai
        from dotenv import load_dotenv
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("正在安装依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_config():
    """检查配置文件"""
    print(os.path)
    if not os.path.exists('.env') and not os.path.exists('.streamlit/secrets.toml'):
        print("⚠️  警告: 未找到配置文件 (.env 或 .streamlit/secrets.toml)")
        print("请确保已正确配置Azure OpenAI的API密钥和端点")

def main():
    print("🚀 启动Azure OpenAI聊天助手...")
    check_dependencies()
    check_config()
    
    # 启动Streamlit应用
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
