# 🔍 NeuralUnblur

NeuralUnblur is an end-to-end deep learning application for image restoration and deblurring. It combines a lightweight Residual CNN (`DeblurNet`), tiled inference for high-resolution processing, and a real-time interactive web interface powered by FastAPI.

---

## ✨ Features

- **Deep Learning Deblurring**: Custom ResNet-style convolutional architecture (`DeblurNet`) trained with mixed precision (AMP) and cosine annealing scheduler.
- **Tiled Inference**: Seamless sliding-window tiled processing with blended overlapping borders to handle arbitrary image resolutions without running out of GPU VRAM.
- **Interactive Web UI**: Clean, responsive browser interface featuring side-by-side comparison slider, drag-and-drop file upload, test blur simulator, and direct image download.
- **FastAPI REST API**: Fully documented asynchronous endpoints with Swagger UI at `/docs`.
- **Synthetic Blur Pipeline**: Supports both Gaussian blur and directional motion blur generation for dataset creation and live testing.

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Torchvision, mixed precision (`torch.amp`)
- **Backend & API**: FastAPI, Uvicorn, Python-Multipart
- **Computer Vision & Processing**: OpenCV, Pillow, NumPy
- **Frontend**: Vanilla HTML5, CSS3, modern JavaScript (no external build step required)

---

## 📁 Project Structure

```text
NeuralUnblur/
├── backend.py            # FastAPI server, training pipeline & model architecture
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration (ignored in git)
├── env.example           # Example configuration template
├── .gitignore            # Git ignore rules
├── templates/
│   └── index.html        # Interactive web UI
├── checkpoints/          # Saved model weights (e.g. deblurnet.pt, ignored in git)
└── data/                 # CIFAR-10 / custom training datasets (ignored in git)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/NeuralUnblur.git
cd NeuralUnblur
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for CUDA GPU acceleration**: To use your NVIDIA GPU, ensure you install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp env.example .env
```

*(On Windows PowerShell, use `Copy-Item env.example .env`)*

Adjust parameters in `.env` if desired (defaults work out of the box).

---

## 🏋️ Training the Model

To train the deblurring network on CIFAR-10 (automatically downloaded on first run):

```bash
python backend.py train
```

- **Device**: Automatically detects CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU.
- **Checkpoints**: The best model based on validation PSNR is automatically saved to `checkpoints/deblurnet.pt`.

---

## 🖥️ Running the Application

Launch the server:

```bash
python backend.py serve
```

Once running:
- **Web Interface**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Interactive API Docs (Swagger)**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the Web UI |
| `GET` | `/api` | API status and model load state |
| `GET` | `/health` | System health, device info, and model metadata |
| `POST` | `/deblur` | Upload an image to receive the deblurred output |
| `POST` | `/blur-demo` | Apply synthetic Gaussian or motion blur to test |
| `POST` | `/train` | Trigger background model training (requires admin token if configured) |
| `GET` | `/train/status` | Current training progress and metrics |

---

## ⚙️ Configuration Reference

Key variables configurable in `.env`:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `MODEL_PATH` | `checkpoints/deblurnet.pt` | Path to save/load model weights |
| `DEVICE` | `auto` | Compute device (`auto`, `cuda`, `mps`, `cpu`) |
| `TILE_SIZE` | `384` | Tile dimension for high-res inference |
| `EPOCHS` | `15` | Number of training epochs |
| `BATCH_SIZE` | `64` | Training batch size |
| `TRAIN_SUBSET` | `20000` | Number of images to use from dataset |

---

## 📄 License

This project is licensed under the MIT License.
