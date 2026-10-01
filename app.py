import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="VisionScript", layout="wide")

# ─────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, #1a1f4a, #05070f 70%);
    color: white;
}

/* HERO */
.hero {
    text-align: center;
    padding: 80px 20px 40px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg,#c084fc,#60a5fa,#34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #9ca3af;
    margin-top: 10px;
}

/* PILLS */
.pill {
    display: inline-block;
    padding: 6px 14px;
    margin: 5px;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    font-size: 12px;
}

/* MAIN CARDS */
.card {
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* IMAGE SIZE FIX */
.image-box img {
    max-height: 250px;
    object-fit: contain;
    border-radius: 12px;
}

/* CAPTION */
.caption-box {
    background: rgba(192,132,252,0.08);
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
}

/* INFO SECTION */
.info-section {
    margin-top: 60px;
    padding: 0 40px 60px;
}

.info-card {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
    min-height: 180px;
    transition: 0.3s;
}

.info-card:hover {
    transform: translateY(-6px);
    border: 1px solid rgba(192,132,252,0.5);
}

.info-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
}

.info-desc {
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.6;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#c084fc,#60a5fa);
    border-radius: 10px;
    height: 45px;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">
        Turn Any Image Into<br>Descriptive Language
    </div>
    <div class="hero-sub">
        Upload a photo and generate captions using AI
    </div>
    <div style="margin-top:20px;">
        <span class="pill">🧠 ResNet-50</span>
        <span class="pill">🎯 Attention</span>
        <span class="pill">✍️ LSTM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
vocab = pickle.load(open("vocab_attention.pkl", "rb"))
word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(2048+256,256)
        self.v = nn.Linear(256,1)

    def forward(self, features, hidden):
        hidden = hidden.unsqueeze(1).repeat(1, features.size(1),1)
        energy = torch.tanh(self.attn(torch.cat((features,hidden),dim=2)))
        attn = self.v(energy).squeeze(2)
        alpha = torch.softmax(attn,dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,256)
        self.attention = Attention()
        self.lstm = nn.LSTMCell(256+2048,256)
        self.fc = nn.Linear(256,vocab_size)

model = Decoder(len(word2idx))
model.load_state_dict(torch.load("attention_model.pth", map_location="cpu"))
model.eval()

# ─────────────────────────────────────────────
# RESNET
# ─────────────────────────────────────────────
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-2])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────
def extract_features(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(image)
    features = features.view(2048,-1).permute(1,0)
    return features

def generate_caption(feature):
    feature = feature.unsqueeze(0)
    h = torch.zeros(1,256)
    c = torch.zeros(1,256)

    word = word2idx["<start>"]
    result = []

    for _ in range(20):
        inp = torch.tensor([word])
        emb = model.embed(inp)
        context = model.attention(feature,h)

        x = torch.cat((emb,context),dim=1)
        h,c = model.lstm(x,(h,c))
        out = model.fc(h)

        word = out.argmax(1).item()

        if word == word2idx["<end>"]:
            break

        result.append(idx2word.get(word,""))

    return " ".join(result)

# ─────────────────────────────────────────────
# MAIN SECTION
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="card image-box">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.button("Generate Caption"):
            with st.spinner("Generating..."):
                features = extract_features(image)
                caption = generate_caption(features)

                st.markdown(
                    f'<div class="caption-box">{caption}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODELS TITLE
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 0 40px; margin-top:40px;">
    <h2 style="color:white; font-size:28px; font-weight:700;">🧠 Models Used</h2>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#c084fc,transparent);margin:15px 0 25px;"></div>
</div>
""", unsafe_allow_html=True)

# MODEL INFO
st.markdown('<div class="info-section">', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">🧠 ResNet-50 Encoder</div>
        <div class="info-desc">
            Extracts deep visual features from images, capturing objects, textures,
            and spatial relationships using convolutional layers.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">🎯 Attention Mechanism</div>
        <div class="info-desc">
            Dynamically focuses on important image regions while generating each word.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">✍️ LSTM Decoder</div>
        <div class="info-desc">
            Generates captions word-by-word while maintaining context and grammar.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APPLICATIONS TITLE
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 0 40px; margin-top:40px;">
    <h2 style="color:white; font-size:28px; font-weight:700;">🚀 Applications</h2>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#60a5fa,transparent);margin:15px 0 25px;"></div>
</div>
""", unsafe_allow_html=True)

# APPLICATIONS
st.markdown('<div class="info-section">', unsafe_allow_html=True)

a1, a2, a3 = st.columns(3)

with a1:
    st.markdown("""
    <div class="info-card">
        <b>👁️ Assistive Technology</b><br><br>
        Helps visually impaired users understand images through captions.
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="info-card">
        <b>📱 Social Media</b><br><br>
        Automatically generates captions for posts and content.
    </div>
    """, unsafe_allow_html=True)

with a3:
    st.markdown("""
    <div class="info-card">
        <b>🔍 Image Search</b><br><br>
        Converts images into searchable text descriptions.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)