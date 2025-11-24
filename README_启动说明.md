# Kindle RAG 智能问答系统 - 启动指南

## 使用 Anaconda 运行

### 方法一：使用 Anaconda Prompt（推荐）

1. **打开 Anaconda Prompt**
   - 在开始菜单搜索 "Anaconda Prompt" 并打开

2. **进入项目目录**
   ```bash
   cd C:\Users\JCJaaa\Desktop\NLP\kindle_agent
   ```

3. **创建 conda 环境**
   ```bash
   conda env create -f environment.yml
   ```

4. **激活环境**
   ```bash
   conda activate kindle_agent
   ```

5. **启动应用**
   
   **方式 A：Flask（带 Web 界面）**
   ```bash
   python app.py
   ```
   然后访问：http://localhost:5000
   
   **方式 B：FastAPI（仅 API）**
   ```bash
   uvicorn api:app --reload --port 8000
   ```
   然后访问：http://localhost:8000

---

### 方法二：使用批处理脚本（快速启动）

直接双击运行：
- `start_flask.bat` - 启动 Flask 应用（带 Web 界面）
- `start_fastapi.bat` - 启动 FastAPI 应用（仅 API）

**注意**：如果 conda 环境未创建，脚本会自动使用当前 Python 环境。

---

### 方法三：手动创建 conda 环境

如果 `environment.yml` 不工作，可以手动创建：

```bash
# 创建环境
conda create -n kindle_agent python=3.11

# 激活环境
conda activate kindle_agent

# 安装依赖
pip install -r requirements.txt
```

---

## 首次运行说明

1. **索引构建**：首次运行时会自动构建向量索引（需要几分钟）
2. **API 密钥**：已在 `rag_backend.py` 中配置，无需修改
3. **PDF 文件**：确保 `Kindle_User's_Guide_English.pdf` 在项目根目录

---

## 常见问题

### Q: conda 命令找不到？
A: 请使用 **Anaconda Prompt** 而不是普通命令行，或者将 Anaconda 添加到系统 PATH。

### Q: 依赖安装失败？
A: 尝试使用国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: faiss-cpu 安装失败？
A: 可以尝试：
```bash
conda install -c conda-forge faiss-cpu
```

---

## 访问地址

- **Flask Web 界面**：http://localhost:5000
- **FastAPI API**：http://localhost:8000
- **FastAPI 文档**：http://localhost:8000/docs

