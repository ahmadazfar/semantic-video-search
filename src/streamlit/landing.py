import streamlit as st

def render_landing_page():
    """Render the landing / home page for Semantic Video Search."""

    # ── Hero Section ──────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        .hero-container {
            text-align: center;
            padding: 3rem 1rem 2rem 1rem;
        }
        .hero-title {
            font-family: 'Inter', sans-serif;
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.25rem;
            color: #9ca3af;
            max-width: 700px;
            margin: 0 auto 2rem auto;
            line-height: 1.6;
        }
        .badge {
            display: inline-block;
            background: rgba(102, 126, 234, 0.15);
            color: #667eea;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
            letter-spacing: 0.03em;
        }

        /* ── Feature cards ────────────────────────────────────── */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin: 1rem 0 2.5rem 0;
        }
        @media (max-width: 800px) {
            .features-grid { grid-template-columns: 1fr; }
        }
        .feature-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 2rem 1.5rem;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-4px);
            border-color: rgba(102,126,234,0.4);
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
        }
        .feature-title {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.15rem;
            color: #e5e7eb;
            margin-bottom: 0.5rem;
        }
        .feature-desc {
            font-family: 'Inter', sans-serif;
            font-size: 0.92rem;
            color: #9ca3af;
            line-height: 1.55;
        }

        /* ── Pipeline section ─────────────────────────────────── */
        .section-header {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.6rem;
            color: #e5e7eb;
            text-align: center;
            margin: 2.5rem 0 0.5rem 0;
        }
        .section-sub {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: #9ca3af;
            text-align: center;
            margin-bottom: 2rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }

        /* ── Pipeline steps ───────────────────────────────────── */
        .pipeline-row {
            display: flex;
            align-items: stretch;
            gap: 0;
            margin: 0 auto 2.5rem auto;
            max-width: 1000px;
            justify-content: center;
        }
        .pipeline-step {
            flex: 1;
            text-align: center;
            padding: 1.5rem 1rem;
            position: relative;
            max-width: 220px;
        }
        .pipeline-step::after {
            content: '→';
            position: absolute;
            right: -12px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.4rem;
            color: #667eea;
            font-weight: bold;
        }
        .pipeline-step:last-child::after {
            content: '';
        }
        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.6rem;
        }
        .step-title {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: #e5e7eb;
            margin-bottom: 0.3rem;
        }
        .step-desc {
            font-family: 'Inter', sans-serif;
            font-size: 0.82rem;
            color: #9ca3af;
            line-height: 1.45;
        }

        /* ── Tech stack pills ─────────────────────────────────── */
        .tech-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            justify-content: center;
            margin: 1rem 0 2rem 0;
        }
        .tech-pill {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 999px;
            padding: 0.45rem 1.1rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #d1d5db;
            font-weight: 500;
            white-space: nowrap;
        }
        .tech-pill .tp-icon {
            margin-right: 0.35rem;
        }

        /* ── Architecture diagram container ───────────────────── */
        .arch-container {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0 2rem 0;
        }

        /* ── Stats row ────────────────────────────────────────── */
        .stats-row {
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        .stat-item {
            text-align: center;
        }
        .stat-num {
            font-family: 'Inter', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #9ca3af;
            margin-top: 0.15rem;
        }

        /* ── Divider ──────────────────────────────────────────── */
        .fancy-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
            margin: 2.5rem 0;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero
    st.markdown(
        """
        <div class="hero-container">
            <div class="badge">AI-Powered Video Intelligence</div>
            <div class="hero-title">Semantic Video Search</div>
            <div class="hero-subtitle">
                Find any object in any video using natural language.
                Upload, index, and search — powered by RF-DETR, CLIP, and ChromaDB.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Feature Cards ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">What It Does</div>
        <div class="section-sub">Semantic video search from uploaded videos to find whatever you are looking for.</div>

        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🎥</div>
                <div class="feature-title">Upload & Index</div>
                <div class="feature-desc">
                    Drop a video and the system automatically detects, tracks, and embeds every object with persistent IDs.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Natural Language Search</div>
                <div class="feature-desc">
                    Type a query like <em>"red truck"</em> or <em>"person with blue backpack"</em> to find matching timestamps from videos selected.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title"> Annotated Output</div>
                <div class="feature-desc">
                    Get an annotated video clip highlighting only the objects that best match your query with similarity scores.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── How It Works (Pipeline) ───────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">How It Works</div>
        <div class="section-sub">Two pipelines — one to index, one to search.</div>
        """,
        unsafe_allow_html=True,
    )

    # Indexing pipeline
    st.markdown("##### 📥 Indexing Pipeline")
    st.markdown(
        """
        <div class="pipeline-row">
            <div class="pipeline-step">
                <div class="step-num">1</div>
                <div class="step-title">Upload Video</div>
                <div class="step-desc">Video is saved and clipped to a manageable segment</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">2</div>
                <div class="step-title">Detect Objects</div>
                <div class="step-desc">RF-DETR runs every 5th frame; optical flow fills in gaps</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">3</div>
                <div class="step-title">Track & Crop</div>
                <div class="step-desc">ByteTrack assigns persistent IDs; objects are cropped 224×224</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">4</div>
                <div class="step-title">Embed & Store</div>
                <div class="step-desc">CLIP encodes crops; averaged vectors stored in ChromaDB</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search pipeline
    st.markdown("##### 🔍 Search Pipeline")
    st.markdown(
        """
        <div class="pipeline-row">
            <div class="pipeline-step">
                <div class="step-num">1</div>
                <div class="step-title">Text Query</div>
                <div class="step-desc">Your natural-language query is encoded by CLIP</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">2</div>
                <div class="step-title">Vector Search</div>
                <div class="step-desc">ChromaDB finds the closest object embeddings via cosine sim</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">3</div>
                <div class="step-title">Re-Detect</div>
                <div class="step-desc">RF-DETR re-runs on the matched timestamp range</div>
            </div>
            <div class="pipeline-step">
                <div class="step-num">4</div>
                <div class="step-title">Annotate</div>
                <div class="step-desc">Top-K matching objects are highlighted in the output video</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">Tech Stack</div>
        <div class="section-sub">Built with state-of-the-art open-source models and tools.</div>
        """,
        unsafe_allow_html=True,
    )

    tech_items = [
        ("🔬", "RF-DETR Medium"),
        ("👁️", "CLIP ViT-B/16"),
        ("🎯", "ByteTrack"),
        ("📐", "Optical Flow (LK)"),
        ("🗄️", "ChromaDB"),
        ("📹", "OpenCV"),
        ("🖥️", "Streamlit"),
        ("🐍", "Python"),
        ("⚡", "CUDA / GPU"),
        ("🧠", "Video-LLaVA 7B"),
    ]
    pills_html = "".join(
        f'<span class="tech-pill"><span class="tp-icon">{icon}</span>{name}</span>'
        for icon, name in tech_items
    )
    st.markdown(f'<div class="tech-grid">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)


    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; color:#6b7280; font-size:0.82rem; margin-top:3rem; padding-bottom:1.5rem;">
            Video Search AI &nbsp;·&nbsp; Semantic Object Detection & Tracking
        </div>
        """,
        unsafe_allow_html=True,
    )
