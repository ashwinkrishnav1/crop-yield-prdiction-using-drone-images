"""
generate_powerpoint.py
Builds the complete, colorful, professional 19-slide MaizeSight PowerPoint presentation (.pptx)
with embedded high-resolution figures, comparison charts, ablation tables, badges,
and comprehensive speaker notes.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    DARK_BG = RGBColor(11, 32, 25)          # Deep Forest Green #0B2019
    DARK_CARD = RGBColor(18, 48, 38)        # Dark Card #123026
    DARK_CARD_BORDER = RGBColor(38, 85, 68) # Card border
    LIGHT_BG = RGBColor(248, 250, 248)      # Light canvas #F8FAF8
    WHITE = RGBColor(255, 255, 255)
    FOREST = RGBColor(15, 45, 36)           # Header Forest Green #0F2D24
    GOLD = RGBColor(229, 169, 60)           # Maize Gold #E5A93C
    GOLD_LIGHT = RGBColor(254, 243, 199)    # Light gold pill
    EMERALD = RGBColor(46, 125, 50)         # Leaf green #2E7D32
    EMERALD_LIGHT = RGBColor(232, 245, 233) # Pale emerald pill
    SLATE_DARK = RGBColor(24, 38, 32)
    SLATE_MUTED = RGBColor(85, 105, 95)
    BORDER_LIGHT = RGBColor(226, 232, 240)
    BORDER_EMERALD = RGBColor(165, 214, 167)
    BORDER_GOLD = RGBColor(246, 210, 143)
    CORAL = RGBColor(220, 38, 38)
    CORAL_LIGHT = RGBColor(254, 242, 242)

    FONT_MAIN = "Calibri"

    def bg(slide, color):
        s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        s.fill.solid()
        s.fill.fore_color.rgb = color
        s.line.fill.background()
        return s

    def add_header(slide, badge, title, subtitle=None, is_dark=False):
        # Badge Pill
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.35), Inches(4.2), Inches(0.32))
        pill.fill.solid()
        pill.fill.fore_color.rgb = GOLD if is_dark else EMERALD_LIGHT
        pill.line.color.rgb = GOLD if is_dark else EMERALD
        pill.line.width = Pt(1)
        tf_pill = pill.text_frame
        tf_pill.word_wrap = True
        p_pill = tf_pill.paragraphs[0]
        p_pill.text = badge.upper()
        p_pill.font.name = FONT_MAIN
        p_pill.font.size = Pt(9.5)
        p_pill.font.bold = True
        p_pill.font.color.rgb = DARK_BG if is_dark else EMERALD
        p_pill.alignment = PP_ALIGN.CENTER

        # Title & Subtitle Box
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.92))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_MAIN
        p_t.font.size = Pt(21)
        p_t.font.bold = True
        p_t.font.color.rgb = WHITE if is_dark else FOREST

        if subtitle:
            p_s = tf.add_paragraph()
            p_s.text = subtitle
            p_s.font.name = FONT_MAIN
            p_s.font.size = Pt(11.5)
            p_s.font.color.rgb = RGBColor(175, 205, 192) if is_dark else SLATE_MUTED
            p_s.space_before = Pt(3)

    def card(slide, left, top, width, height, bg_col=WHITE, border_col=BORDER_LIGHT, b_width=1):
        c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        c.fill.solid()
        c.fill.fore_color.rgb = bg_col
        if border_col:
            c.line.color.rgb = border_col
            c.line.width = Pt(b_width)
        else:
            c.line.fill.background()
        return c

    def add_notes(slide, text):
        slide.notes_slide.notes_text_frame.text = text

    # ==========================================================
    # SLIDE 1: TITLE SLIDE (Rich Dark Maize Theme)
    # ==========================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg(s1, DARK_BG)

    # Decorative Gold Top Banner
    bar_top = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
    bar_top.fill.solid()
    bar_top.fill.fore_color.rgb = GOLD
    bar_top.line.fill.background()

    # Badges container
    b_tag = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.85), Inches(5.6), Inches(0.38))
    b_tag.fill.solid()
    b_tag.fill.fore_color.rgb = DARK_CARD
    b_tag.line.color.rgb = GOLD
    b_tag.line.width = Pt(1)
    tf_b = b_tag.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.text = "UAV REMOTE SENSING  |  DEEP VISION  |  CROSS-SITE ML"
    p_b.font.name = FONT_MAIN
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = GOLD
    p_b.alignment = PP_ALIGN.CENTER

    # Main Title
    t_box1 = s1.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(7.4), Inches(2.3))
    tf1 = t_box1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0
    p1 = tf1.paragraphs[0]
    p1.text = "MaizeSight"
    p1.font.name = FONT_MAIN
    p1.font.size = Pt(46)
    p1.font.bold = True
    p1.font.color.rgb = GOLD

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Multimodal UAV Remote Sensing & Machine Learning for In-Season Maize Yield Prediction Across the US Corn Belt"
    p1_sub.font.name = FONT_MAIN
    p1_sub.font.size = Pt(17)
    p1_sub.font.bold = True
    p1_sub.font.color.rgb = WHITE
    p1_sub.space_before = Pt(6)

    p1_sub2 = tf1.add_paragraph()
    p1_sub2.text = "High-Resolution Drone Imagery • EfficientNet-B0 Latent Embeddings • Biophysical Canopy Indices • Decoupled Cross-Site Generalization"
    p1_sub2.font.name = FONT_MAIN
    p1_sub2.font.size = Pt(11)
    p1_sub2.font.color.rgb = RGBColor(170, 200, 185)
    p1_sub2.space_before = Pt(6)

    # Left feature metrics card
    feat_card = card(s1, Inches(0.8), Inches(3.9), Inches(7.4), Inches(2.95), bg_col=DARK_CARD, border_col=DARK_CARD_BORDER)
    tf_f = feat_card.text_frame
    tf_f.word_wrap = True
    tf_f.margin_top = Inches(0.18)
    tf_f.margin_left = Inches(0.25)
    tf_f.margin_right = Inches(0.25)
    
    p_f0 = tf_f.paragraphs[0]
    p_f0.text = "KEY PROJECT HIGHLIGHTS & ARCHITECTURAL ADVANCES"
    p_f0.font.bold = True
    p_f0.font.size = Pt(11)
    p_f0.font.color.rgb = GOLD

    highlights = [
        ("Multi-Temporal Triad:", "Tracks emergence (TP1), canopy closure (TP2), and flowering (TP3)."),
        ("Dual-Model Strategy:", "Model 1 In-Domain Specialist (R² = 0.845) vs Model 2 Zero-Shot Generalist (LOLO)."),
        ("Audited Dataset:", "2,131 verified plots across 5 trial sites in Nebraska, Iowa, and Indiana (Dryad)."),
        ("Overcoming Domain Shift:", "Decoupled Field Anchor + Within-Field Ranker flips negative R² to +0.217.")
    ]
    for b_prefix, b_text in highlights:
        p = tf_f.add_paragraph()
        p.space_before = Pt(7)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = WHITE
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = RGBColor(190, 215, 205)

    # Right Image Embed (USA Map)
    map_path = "project__imagesss/USA Highlighted Midwest States Map.png"
    if os.path.exists(map_path):
        s1.shapes.add_picture(map_path, Inches(8.5), Inches(1.15), width=Inches(4.1), height=Inches(4.1))
        # Image caption badge below
        c_tag = card(s1, Inches(8.5), Inches(5.45), Inches(4.1), Inches(1.4), bg_col=DARK_CARD, border_col=DARK_CARD_BORDER)
        tf_c = c_tag.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = Inches(0.18)
        p_c1 = tf_c.paragraphs[0]
        p_c1.text = "RESEARCH GEOGRAPHY (2022 SEASON)"
        p_c1.font.bold = True
        p_c1.font.size = Pt(9.5)
        p_c1.font.color.rgb = GOLD
        p_c2 = tf_c.add_paragraph()
        p_c2.text = "5 Multi-State Research Sites: Scottsbluff NE, Lincoln NE, Ames IA, Missouri Valley IA, and Crawfordsville IN."
        p_c2.font.size = Pt(9)
        p_c2.font.color.rgb = RGBColor(190, 215, 205)
        p_c2.space_before = Pt(3)

    add_notes(s1, "Welcome to the MaizeSight presentation. Today we will present a complete multimodal remote sensing and machine learning system that accurately predicts maize grain yield weeks before harvest. By combining high-resolution UAV drone imagery at 3 key life-cycle stages with deep learning and tabular gradient boosting, we solve one of precision agriculture's hardest challenges: generalizing across unseen geographic fields.")

    # ==========================================================
    # SLIDE 2: INTRODUCTION & MOTIVATION (Section i)
    # ==========================================================
    s2 = prs.slides.add_slide(blank_layout)
    bg(s2, LIGHT_BG)
    add_header(s2, "Section i • Introduction & Motivation", "The Agricultural Imperative: In-Season Yield Intelligence", "Why real-time, non-destructive crop monitoring drives economic and food security")

    col_w = Inches(3.68)
    top_pos = Inches(1.8)
    h_pos = Inches(5.2)

    # Card 1: Food Security & Economic Scale
    c1 = card(s2, Inches(0.8), top_pos, col_w, h_pos, bg_col=WHITE, border_col=BORDER_LIGHT)
    tf1 = c1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_right = tf1.margin_top = Inches(0.25)
    p = tf1.paragraphs[0]
    p.text = "1. Global Food Security & Scale"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = FOREST

    points_c1 = [
        ("Cornerstone Crop:", "Maize (Zea mays) is a vital global commodity supplying animal feed, human sustenance, and industrial biofuels."),
        ("Economic Scale:", "Hundreds of millions of bushels are produced each year across the US Corn Belt, driving billions in agricultural trade."),
        ("Climate Volatility:", "Increasingly erratic seasonal weather patterns (flash droughts, heatwaves, extreme rain) make early yield estimation critical for supply-chain resilience.")
    ]
    for b_prefix, b_text in points_c1:
        p = tf1.add_paragraph()
        p.space_before = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = FOREST
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(10)
        r2.font.color.rgb = SLATE_DARK

    # Card 2: Limitations of Existing Approaches
    c2 = card(s2, Inches(4.82), top_pos, col_w, h_pos, bg_col=WHITE, border_col=BORDER_LIGHT)
    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = tf2.margin_top = Inches(0.25)
    p = tf2.paragraphs[0]
    p.text = "2. Flaws of Conventional Methods"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = CORAL

    points_c2 = [
        ("Manual Field Sampling:", "Labor-intensive, destructive to plants, and too sparse to capture spatial micro-variability across large plots."),
        ("Combine Yield Monitors:", "Provide high-precision yield data, but only post-harvest. That is far too late for mid-season nitrogen or irrigation management."),
        ("Satellite Remote Sensing:", "Coarse spatial resolution (10m–30m in Sentinel/Landsat) blends multiple rows; frequent cloud obscuration during key flowering windows.")
    ]
    for b_prefix, b_text in points_c2:
        p = tf2.add_paragraph()
        p.space_before = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = CORAL
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(10)
        r2.font.color.rgb = SLATE_DARK

    # Card 3: The UAV Drone Advantage
    c3 = card(s2, Inches(8.84), top_pos, col_w, h_pos, bg_col=WHITE, border_col=BORDER_EMERALD, b_width=1.5)
    tf3 = c3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_right = tf3.margin_top = Inches(0.25)
    p = tf3.paragraphs[0]
    p.text = "3. The UAV Remote Sensing Solution"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = EMERALD

    points_c3 = [
        ("Sub-Centimeter Resolution:", "Resolves individual maize leaves, row margins, tasseling heads, and inter-row weed/canopy gaps."),
        ("On-Demand Temporal Agility:", "Fly under cloud ceilings precisely at critical biological stages: seedling emergence, canopy closure, and flowering."),
        ("Non-Destructive Tracking:", "Monitors identical plots continuously across the full season to capture temporal growth trajectories.")
    ]
    for b_prefix, b_text in points_c3:
        p = tf3.add_paragraph()
        p.space_before = Pt(10)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = EMERALD
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(10)
        r2.font.color.rgb = SLATE_DARK

    add_notes(s2, "Slide 2 outlines the motivation behind MaizeSight. Maize is crucial for global nutrition and energy. However, farmers and agronomists have historically been constrained: manual testing is destructive and slow, satellites suffer from coarse resolution and clouds, and combine harvesters only tell you the yield when the season is already over. Low-altitude drone flights solve this by providing centimeter-level imagery exactly when corrective actions can still be taken.")

    # ==========================================================
    # SLIDE 3: PROBLEM STATEMENT & REAL-WORLD RELEVANCE (Section ii)
    # ==========================================================
    s3 = prs.slides.add_slide(blank_layout)
    bg(s3, LIGHT_BG)
    add_header(s3, "Section ii • Problem Statement & Relevance", "The Cross-Site Generalization Trap in Precision Agriculture", "Why computer vision models fail catastrophically when transferred to new geographic fields")

    w_card = Inches(5.7)
    left_c = card(s3, Inches(0.8), Inches(1.8), w_card, Inches(5.2), bg_col=WHITE, border_col=BORDER_LIGHT)
    tf_lc = left_c.text_frame
    tf_lc.word_wrap = True
    tf_lc.margin_left = tf_lc.margin_right = tf_lc.margin_top = Inches(0.25)
    
    p = tf_lc.paragraphs[0]
    p.text = "1. The Technical Challenge: Environmental Domain Shift"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = FOREST

    points_prob = [
        ("Complex G x E x M Biology:", "Crop yield is not a pure function of canopy greenness. It is governed by non-linear Genotype x Environment x Management interactions."),
        ("Environmental Baseline Offsets:", "Different geographic sites have different soil types (loam vs alluvial), planting dates, and weather shocks. A lush canopy in a drought site yields very differently from one in prime soil."),
        ("The Negative R² Collapse:", "Standard deep neural networks trained on one farm achieve R² > 0.80 locally, but when evaluated on an unseen farm, test R² collapses to -1.62 (worse than guessing the field mean)."),
        ("Overfitting to Local Artifacts:", "Models easily memorize site-specific soil colors, flight illumination angles, or plot layout patterns rather than generalizable crop physiology.")
    ]
    for b_prefix, b_text in points_prob:
        p = tf_lc.add_paragraph()
        p.space_before = Pt(9)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = FOREST
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(10)
        r2.font.color.rgb = SLATE_DARK

    right_c = card(s3, Inches(6.8), Inches(1.8), w_card, Inches(5.2), bg_col=WHITE, border_col=BORDER_GOLD, b_width=1.5)
    tf_rc = right_c.text_frame
    tf_rc.word_wrap = True
    tf_rc.margin_left = tf_rc.margin_right = tf_rc.margin_top = Inches(0.25)

    p = tf_rc.paragraphs[0]
    p.text = "2. Real-World Relevance & Stakeholder Impact"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = GOLD

    points_rel = [
        ("Commercial Farm Managers:", "Enables variable-rate nitrogen fertilization during mid-season vegetative growth, optimizing grain fill while preventing nitrogen leaching into waterways."),
        ("Plant Breeders & Geneticists:", "Enables rapid phenotypic screening of thousands of experimental hybrid crosses across multi-acre plots without waiting for manual harvest."),
        ("Crop Insurance & Commodity Markets:", "Provides objective, tamper-proof, plot-scale yield estimations to settle drought insurance claims and forecast regional grain supply weeks in advance."),
        ("Logistics & Storage Planning:", "Allows grain elevator operators and shipping fleets to stage railcars and silos based on high-confidence pre-harvest yield maps.")
    ]
    for b_prefix, b_text in points_rel:
        p = tf_rc.add_paragraph()
        p.space_before = Pt(9)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = FOREST
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(10)
        r2.font.color.rgb = SLATE_DARK

    add_notes(s3, "Slide 3 presents the central research problem. In precision agriculture, the real test of machine learning is out-of-domain transfer. When an algorithm trained in Nebraska is tested in Indiana, the baseline yield shifts drastically due to climate and soil differences. Off-the-shelf CNNs score negative R-squared. Solving this problem has massive economic consequences for farmers, breeders, and commodity supply chains.")

    # ==========================================================
    # SLIDE 4: OBJECTIVES & SCOPE (Section iii)
    # ==========================================================
    s4 = prs.slides.add_slide(blank_layout)
    bg(s4, LIGHT_BG)
    add_header(s4, "Section iii • Objectives & Scope", "Project Objectives & Scope: Dual-Paradigm Architecture", "Architecting high in-domain precision alongside zero-shot cross-geography domain transfer")

    card_w = Inches(5.7)
    card_h = Inches(2.1)
    
    # Obj 1
    o1 = card(s4, Inches(0.8), Inches(1.8), card_w, card_h, bg_col=WHITE)
    tfo1 = o1.text_frame
    tfo1.word_wrap = True
    tfo1.margin_left = tfo1.margin_right = tfo1.margin_top = Inches(0.2)
    p = tfo1.paragraphs[0]
    p.text = "1. End-to-End Multimodal Remote Sensing Pipeline"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = FOREST
    p_sub = tfo1.add_paragraph()
    p_sub.text = "Integrate multi-temporal UAV RGB orthomosaics across three phenological stages (TP1, TP2, TP3) with ground-truth agronomic tabular variables (stand count, planting date, pedigree crosses) into a unified computational workflow."
    p_sub.font.size = Pt(10)
    p_sub.font.color.rgb = SLATE_DARK
    p_sub.space_before = Pt(4)

    # Obj 2
    o2 = card(s4, Inches(6.8), Inches(1.8), card_w, card_h, bg_col=WHITE)
    tfo2 = o2.text_frame
    tfo2.word_wrap = True
    tfo2.margin_left = tfo2.margin_right = tfo2.margin_top = Inches(0.2)
    p = tfo2.paragraphs[0]
    p.text = "2. Dual-Model Architectural Strategy"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = EMERALD
    p_sub = tfo2.add_paragraph()
    p_sub.text = "Build two complementary production models: Model 1 (In-Domain Specialist) maximizing precision across monitored research fields, and Model 2 (Zero-Shot Generalist) overcoming site shifts via a decoupled Field Anchor + Within-Field Ranker."
    p_sub.font.size = Pt(10)
    p_sub.font.color.rgb = SLATE_DARK
    p_sub.space_before = Pt(4)

    # Obj 3
    o3 = card(s4, Inches(0.8), Inches(4.1), card_w, card_h, bg_col=WHITE)
    tfo3 = o3.text_frame
    tfo3.word_wrap = True
    tfo3.margin_left = tfo3.margin_right = tfo3.margin_top = Inches(0.2)
    p = tfo3.paragraphs[0]
    p.text = "3. Hybrid Vision-Agronomic Feature Engineering"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = GOLD
    p_sub = tfo3.add_paragraph()
    p_sub.text = "Extract 1,280-D latent deep embeddings via fine-tuned EfficientNet-B0, fuse with 12 biophysical canopy indices (ExG, VARI, FGCC), model multi-temporal growth velocities, and incorporate pedigree GCA/SCA proxy effects alongside 2D spatial kriging."
    p_sub.font.size = Pt(10)
    p_sub.font.color.rgb = SLATE_DARK
    p_sub.space_before = Pt(4)

    # Obj 4
    o4 = card(s4, Inches(6.8), Inches(4.1), card_w, card_h, bg_col=WHITE)
    tfo4 = o4.text_frame
    tfo4.word_wrap = True
    tfo4.margin_left = tfo4.margin_right = tfo4.margin_top = Inches(0.2)
    p = tfo4.paragraphs[0]
    p.text = "4. Production Web Platform with 3D Visualization"
    p.font.bold = True
    p.font.size = Pt(11.5)
    p.font.color.rgb = FOREST
    p_sub = tfo4.add_paragraph()
    p_sub.text = "Deploy a full-stack web application (FastAPI backend + Three.js/GSAP frontend) offering automated multi-temporal image matching, simultaneous dual-model inference, diagnostic quality flags, and sub-millisecond plot database exploration."
    p_sub.font.size = Pt(10)
    p_sub.font.color.rgb = SLATE_DARK
    p_sub.space_before = Pt(4)

    # Scope Summary Ribbon at bottom
    ribbon = card(s4, Inches(0.8), Inches(6.4), Inches(11.7), Inches(0.6), bg_col=EMERALD_LIGHT, border_col=BORDER_EMERALD)
    tf_rib = ribbon.text_frame
    p_rib = tf_rib.paragraphs[0]
    p_rib.text = "PROJECT SCOPE: 5 Midwest Locations (NE, IA, IN)  •  2,131 Verified Research Plots  •  Standardized 15.5% Moisture Grain Yield (bu/ac)"
    p_rib.font.bold = True
    p_rib.font.size = Pt(10.5)
    p_rib.font.color.rgb = FOREST
    p_rib.alignment = PP_ALIGN.CENTER

    add_notes(s4, "Slide 4 defines our objectives and project boundaries. We set out to create an end-to-end multimodal pipeline that connects drone flights to combine-harvested yields. Rather than relying on a single monolithic model, we designed a dual-paradigm architecture: a Specialist model for monitored sites, and a Generalist model for brand-new fields. The scope encompasses 2,131 audited plots across Nebraska, Iowa, and Indiana.")

    # ==========================================================
    # SLIDE 5: DATASET OVERVIEW & MODALITIES
    # ==========================================================
    s5 = prs.slides.add_slide(blank_layout)
    bg(s5, LIGHT_BG)
    add_header(s5, "Dataset Overview • Source, Geography & Modalities", "Multi-State Maize Field Trials from the US Corn Belt", "Audited experimental footprint published by Shrestha et al. (2022) on Dryad Repository")

    # Left Container: Dataset Table
    w_left5 = Inches(6.8)
    c_tbl = card(s5, Inches(0.8), Inches(1.8), w_left5, Inches(5.2), bg_col=WHITE)
    tf_tbl = c_tbl.text_frame
    tf_tbl.word_wrap = True
    tf_tbl.margin_left = tf_tbl.margin_right = tf_tbl.margin_top = Inches(0.2)

    p = tf_tbl.paragraphs[0]
    p.text = "1. Geographic Footprint & Environmental Diversity (2022)"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = FOREST

    p_meta = tf_tbl.add_paragraph()
    p_meta.text = "Academic Source: Shrestha et al. (2022) Dryad Repository (doi:10.5061/dryad.905qftttm)"
    p_meta.font.size = Pt(9.5)
    p_meta.font.color.rgb = SLATE_MUTED
    p_meta.space_before = Pt(3)

    # Insert Table of the 5 sites
    table_shape = s5.shapes.add_table(6, 4, Inches(0.95), Inches(2.65), Inches(6.5), Inches(2.3))
    table = table_shape.table
    table.columns[0].width = Inches(1.6)
    table.columns[1].width = Inches(0.8)
    table.columns[2].width = Inches(1.1)
    table.columns[3].width = Inches(3.0)

    headers = ["Location, State", "Plots", "Yield (bu/ac)", "Environmental Characteristic"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = FOREST
        p_c = cell.text_frame.paragraphs[0]
        p_c.text = h
        p_c.font.bold = True
        p_c.font.size = Pt(9)
        p_c.font.color.rgb = WHITE

    site_data = [
        ("Scottsbluff, NE", "489", "143.4", "High-altitude semi-arid, irrigated"),
        ("Lincoln, NE", "504", "41.2", "Severe drought & heat stress shock"),
        ("Ames, IA", "487", "130.3", "Prime Central Corn Belt loam soil"),
        ("Missouri Valley, IA", "163", "152.5", "River valley fertile alluvial soil"),
        ("Crawfordsville, IN", "488", "165.1", "Held-out unseen test site (high yield)")
    ]
    for row_idx, row_values in enumerate(site_data, start=1):
        for col_idx, val in enumerate(row_values):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if row_idx % 2 == 1 else RGBColor(245, 248, 246)
            p_c = cell.text_frame.paragraphs[0]
            p_c.text = val
            p_c.font.size = Pt(8.5)
            p_c.font.color.rgb = SLATE_DARK
            if col_idx == 2:
                p_c.font.bold = True

    p_bot = tf_tbl.add_paragraph()
    p_bot.space_before = Pt(175)
    p_bot.text = "Audit Note: 7,281 raw drone image files were audited down to 2,131 verified plots across the 5 sites. North Platte, NE was excluded due to published plot mowing/segmentation artifacts."
    p_bot.font.size = Pt(9)
    p_bot.font.color.rgb = SLATE_MUTED

    # Right Container: 2 Modalities
    w_right5 = Inches(4.7)
    c_mod = card(s5, Inches(7.8), Inches(1.8), w_right5, Inches(5.2), bg_col=WHITE, border_col=BORDER_EMERALD)
    tf_mod = c_mod.text_frame
    tf_mod.word_wrap = True
    tf_mod.margin_left = tf_mod.margin_right = tf_mod.margin_top = Inches(0.2)

    p = tf_mod.paragraphs[0]
    p.text = "2. Two Primary Data Modalities"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = EMERALD

    mod_points = [
        ("A. Multi-Temporal Drone Imagery:", "Ultra-high-resolution RGB orthomosaics cropped to individual 4-row plots (.PNG format) captured across 3 growth milestones:"),
        ("• TP1 (Early Vegetative / V4–V6):", "Seedling emergence, plant stand count, early vigor."),
        ("• TP2 (Canopy Closure / V10–V12):", "Peak vegetative biomass, maximum leaf area cover."),
        ("• TP3 (Flowering / R1 Tasseling):", "Anthesis, pollen shed, reproductive potential."),
        ("B. Ground Truth Agronomic CSV:", "Measured combine harvest metrics:"),
        ("• Target (yieldPerAcre):", "Grain yield in bu/ac corrected to 15.5% moisture."),
        ("• Plant Stand (totalStandCount):", "Living plant count in middle two plot rows."),
        ("• Crop Phenology & Genetics:", "GDD to anthesis + hybrid parentage pedigree (female x male inbred lines).")
    ]
    for b_prefix, b_text in mod_points:
        p = tf_mod.add_paragraph()
        p.space_before = Pt(4)
        r1 = p.add_run()
        r1.text = b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = FOREST if "A." in b_prefix or "B." in b_prefix else SLATE_DARK
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(9)
        r2.font.color.rgb = SLATE_DARK

    add_notes(s5, "Slide 5 introduces our data foundation. Sourced from Shrestha et al. on Dryad, the 2022 season covers 5 distinct environments: from severe drought in Lincoln, Nebraska (averaging 41 bu/ac) to prime alluvial river soil in Missouri Valley and Crawfordsville, Indiana (over 165 bu/ac). We combine two modalities: high-resolution RGB drone imagery across 3 flights and ground-truth combine harvest records standardized to 15.5% moisture.")

    # ==========================================================
    # SLIDE 6: DATA PREPROCESSING & REPRESENTATION TECHNIQUES
    # ==========================================================
    s6 = prs.slides.add_slide(blank_layout)
    bg(s6, LIGHT_BG)
    add_header(s6, "Data Engineering • Preprocessing & Representation", "Data Preprocessing & Multi-Space Representation", "Eliminating edge artifacts, preventing data leakage, and constructing rich feature spaces")

    w_card6 = Inches(5.7)

    # Left: Data Preprocessing Techniques
    c_prep = card(s6, Inches(0.8), Inches(1.8), w_card6, Inches(5.2), bg_col=WHITE)
    tf_prep = c_prep.text_frame
    tf_prep.word_wrap = True
    tf_prep.margin_left = tf_prep.margin_right = tf_prep.margin_top = Inches(0.25)
    
    p = tf_prep.paragraphs[0]
    p.text = "1. Data Preprocessing Techniques"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = FOREST

    points_prep = [
        ("Image Quality Audit & Filtering:", "Automated screening for corrupted image headers, all-black segmented tiles, and optical blur. Aligned filenames with harvest records, dropping unmatched rows."),
        ("Replication-Padded Patch Tiling (224x224):", "Variable plot bounding boxes diced into uniform 224x224 patches using cv2.BORDER_REPLICATE. Preserved outer canopy leaves without artificial black zero-padding borders."),
        ("Strict Plot-Level Anti-Leakage Splitting:", "Guaranteed all 3 seasonal flights (TP1, TP2, TP3) for any plot stayed strictly within the same data split. Never trained on TP1 and tested on TP3 of the same plot!"),
        ("Tabular Imputation & Flight-Level Z-Scoring:", "Median imputation for stand counts and planting dates. Normalized solar elevation and cloud illumination shifts via flight-level z-scoring.")
    ]
    for b_prefix, b_text in points_prep:
        p = tf_prep.add_paragraph()
        p.space_before = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = FOREST
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = SLATE_DARK

    # Right: Representation Spaces
    c_rep = card(s6, Inches(6.8), Inches(1.8), w_card6, Inches(5.2), bg_col=WHITE, border_col=BORDER_EMERALD)
    tf_rep = c_rep.text_frame
    tf_rep.word_wrap = True
    tf_rep.margin_left = tf_rep.margin_right = tf_rep.margin_top = Inches(0.25)

    p = tf_rep.paragraphs[0]
    p.text = "2. Four Numerical Feature Representation Spaces"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = EMERALD

    points_rep = [
        ("A. Deep Latent Vision Space (R¹²⁸⁰):", "Fine-tuned EfficientNet-B0 maps each 224x224 patch into a 1,280-D vector. Mean and standard deviation pooling aggregate patches into a plot-level visual fingerprint."),
        ("B. Biophysical Canopy Indices (R¹² per flight):", "Calculates physiological vegetation metrics: Excess Green (ExG = 2G - R - B), VARI, NGRDI, Fractional Green Canopy Cover (FGCC), and yellow tassel senescence fraction."),
        ("C. Multi-Temporal Dynamic Vectors:", "Growth velocity vectors (ΔTP2-TP1, ΔTP3-TP2) measuring the rate of canopy expansion and speed of reproductive maturation."),
        ("D. Genetic & Spatial Representation:", "Parental pedigree deconstruction into GCA/SCA proxy effects, plus 2D spatial kriging surfaces converting range and row coordinates into micro-soil fertility trends.")
    ]
    for b_prefix, b_text in points_rep:
        p = tf_rep.add_paragraph()
        p.space_before = Pt(8)
        r1 = p.add_run()
        r1.text = "• " + b_prefix + " "
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = EMERALD
        r2 = p.add_run()
        r2.text = b_text
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = SLATE_DARK

    add_notes(s6, "Slide 6 details our data engineering. We enforce two critical techniques: first, replication padding prevents artificial zero-gradient borders from throwing off the CNN filters; second, strict plot-level splitting ensures that all three temporal flights for a given physical plot remain in the same partition. Our feature engineering creates four distinct spaces: deep latent embeddings, biophysical indices, growth rate deltas, and spatial/genetic priors.")

    # ==========================================================
    # SLIDE 7: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
    # ==========================================================
    s7 = prs.slides.add_slide(blank_layout)
    bg(s7, LIGHT_BG)
    add_header(s7, "Exploratory Data Analysis • Patterns & Insights", "Exploratory Data Analysis & Empirical Discoveries", "Uncovering severe cross-site heterogeneity, temporal dynamics, and data completeness")

    # Left Container with 4 Cards
    w_left7 = Inches(6.8)
    cw7 = Inches(3.25)
    ch7 = Inches(2.45)

    # Box 1: Cross-Site Yield Bimodality
    b1 = card(s7, Inches(0.8), Inches(1.8), cw7, ch7, bg_col=WHITE)
    tfb1 = b1.text_frame; tfb1.word_wrap = True; tfb1.margin_left = tfb1.margin_right = tfb1.margin_top = Inches(0.18)
    p = tfb1.paragraphs[0]; p.text = "1. Cross-Site Heterogeneity"; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = FOREST
    points_eda1 = [
        ("Lincoln NE (41.2 bu/ac):", "Severe drought shock."),
        ("Crawfordsville IN (165.1 bu/ac):", "High-fertility soil."),
        ("Key Takeaway:", "Global models fail without field baseline anchoring.")
    ]
    for bp, bt in points_eda1:
        p = tfb1.add_paragraph(); p.space_before = Pt(3)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(8.5); r1.font.color.rgb = FOREST
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(8); r2.font.color.rgb = SLATE_DARK

    # Box 2: Image Audit & Completeness
    b2 = card(s7, Inches(4.25), Inches(1.8), cw7, ch7, bg_col=WHITE)
    tfb2 = b2.text_frame; tfb2.word_wrap = True; tfb2.margin_left = tfb2.margin_right = tfb2.margin_top = Inches(0.18)
    p = tfb2.paragraphs[0]; p.text = "2. Data Completeness"; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = EMERALD
    points_eda2 = [
        ("7,281 Raw Files:", "Filtered corrupt & blurred PNGs."),
        ("2,131 Valid Plots:", "Locked across all 3 flights."),
        ("6,393 Images:", "Complete verified multi-temporal sequences.")
    ]
    for bp, bt in points_eda2:
        p = tfb2.add_paragraph(); p.space_before = Pt(3)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(8.5); r1.font.color.rgb = EMERALD
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(8); r2.font.color.rgb = SLATE_DARK

    # Box 3: Multi-Temporal Dynamics
    b3 = card(s7, Inches(0.8), Inches(4.45), cw7, ch7, bg_col=WHITE)
    tfb3 = b3.text_frame; tfb3.word_wrap = True; tfb3.margin_left = tfb3.margin_right = tfb3.margin_top = Inches(0.18)
    p = tfb3.paragraphs[0]; p.text = "3. Temporal Canopy Growth"; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = GOLD
    points_eda3 = [
        ("TP1 Emergence:", "Soil >80%; stand count dictates vigor."),
        ("TP2 Closure:", "Leaf cover >70%; peak ExG & VARI."),
        ("TP3 Flowering:", "Tassel emergence exhibits strongest yield correlation.")
    ]
    for bp, bt in points_eda3:
        p = tfb3.add_paragraph(); p.space_before = Pt(3)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(8.5); r1.font.color.rgb = GOLD
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(8); r2.font.color.rgb = SLATE_DARK

    # Box 4: Patch Tiling & Padding Exploration
    b4 = card(s7, Inches(4.25), Inches(4.45), cw7, ch7, bg_col=WHITE)
    tfb4 = b4.text_frame; tfb4.word_wrap = True; tfb4.margin_left = tfb4.margin_right = tfb4.margin_top = Inches(0.18)
    p = tfb4.paragraphs[0]; p.text = "4. Edge Artifact Analysis"; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = FOREST
    points_eda4 = [
        ("Zero-Padding Artifacts:", "Black borders corrupted CNN edge filters."),
        ("Replication Padding:", "Preserved true canopy margins & leaf density smoothly.")
    ]
    for bp, bt in points_eda4:
        p = tfb4.add_paragraph(); p.space_before = Pt(4)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(8.5); r1.font.color.rgb = FOREST
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(8); r2.font.color.rgb = SLATE_DARK

    # Right: Embedded Audit Dashboard Image
    audit_img = "project__imagesss/UAV Maize Dataset Audit Dashboard.png"
    if os.path.exists(audit_img):
        s7.shapes.add_picture(audit_img, Inches(7.75), Inches(1.8), width=Inches(4.75), height=Inches(5.1))

    add_notes(s7, "Slide 7 shows key insights from our exploratory data analysis. The yield distribution showed extreme bimodality: Lincoln averaged only 41 bushels per acre due to drought, whereas Crawfordsville reached 165 bu/ac. This proved that a standard model cannot simply guess yields without knowing the field's baseline. We also observed how correlation changes over time: TP1 reflects stand count, TP2 reflects vegetative canopy, and TP3 tasseling shows the highest single-flight correlation to grain yield.")

    # ==========================================================
    # SLIDE 8: DEDICATED SLIDE: HOW CNN WORKS & ROLE IN MAIZESIGHT (VISUAL INFOGRAPHIC)
    # ==========================================================
    s8 = prs.slides.add_slide(blank_layout)
    bg(s8, LIGHT_BG)
    add_header(s8, "Deep Learning Mechanisms • Computer Vision", "How CNNs Work & Their Role in MaizeSight", "Spatial feature extraction, hierarchical feature learning, and fine-tuned EfficientNet-B0")

    cnn_diagram_path = "project__imagesss/cnn_pipeline_diagram.jpg"
    if os.path.exists(cnn_diagram_path):
        # Embed the generated infographic visual diagram directly
        s8.shapes.add_picture(cnn_diagram_path, Inches(0.8), Inches(1.8), width=Inches(11.73), height=Inches(5.15))
    else:
        w_card8 = Inches(5.7)
        # Left: How CNN Works
        c_cnn1 = card(s8, Inches(0.8), Inches(1.8), w_card8, Inches(5.2), bg_col=WHITE)
        tf_cnn1 = c_cnn1.text_frame
        tf_cnn1.word_wrap = True
        tf_cnn1.margin_left = tf_cnn1.margin_right = tf_cnn1.margin_top = Inches(0.25)
        p = tf_cnn1.paragraphs[0]; p.text = "1. How a CNN Works (The Core Mechanism)"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = FOREST
        for b_prefix, b_text in points_cnn_core:
            p = tf_cnn1.add_paragraph(); p.space_before = Pt(8)
            r1 = p.add_run(); r1.text = b_prefix + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = FOREST
            r2 = p.add_run(); r2.text = b_text; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

        # Right: Role in MaizeSight
        c_cnn2 = card(s8, Inches(6.8), Inches(1.8), w_card8, Inches(5.2), bg_col=WHITE, border_col=BORDER_EMERALD)
        tf_cnn2 = c_cnn2.text_frame
        tf_cnn2.word_wrap = True
        tf_cnn2.margin_left = tf_cnn2.margin_right = tf_cnn2.margin_top = Inches(0.25)
        p = tf_cnn2.paragraphs[0]; p.text = "2. The Role of the CNN in MaizeSight"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = EMERALD
        for b_prefix, b_text in points_cnn_role:
            p = tf_cnn2.add_paragraph(); p.space_before = Pt(8)
            r1 = p.add_run(); r1.text = b_prefix + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = EMERALD
            r2 = p.add_run(); r2.text = b_text; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

    add_notes(s8, "Slide 8 focuses specifically on Convolutional Neural Networks. Standard neural networks flatten an image and lose all row geometry. CNNs slide small weight kernels across the 2D image, learning hierarchical representations from simple leaf edges up to complex tassel clusters. In MaizeSight, we use EfficientNet-B0 as a feature extractor: we remove its classification head and output a 1,280-dimensional latent embedding for each patch, pooling them across plots to feed our downstream regressors.")

    # ==========================================================
    # SLIDE 9: DEDICATED SLIDE: HOW XGBOOST WORKS & ROLE IN MAIZESIGHT
    # ==========================================================
    # ==========================================================
    # SLIDE 9: DEDICATED SLIDE: HOW XGBOOST WORKS & ROLE IN MAIZESIGHT (VISUAL INFOGRAPHIC)
    # ==========================================================
    s9 = prs.slides.add_slide(blank_layout)
    bg(s9, LIGHT_BG)
    add_header(s9, "Machine Learning Mechanisms • Gradient Boosting", "How XGBoost Works & Its Role in MaizeSight", "Second-order Taylor optimization, regularized ensembles, and multimodal tabular fusion")

    xgb_diagram_path = "project__imagesss/xgboost_pipeline_diagram.jpg"
    if os.path.exists(xgb_diagram_path):
        # Embed the generated infographic visual diagram directly
        s9.shapes.add_picture(xgb_diagram_path, Inches(0.8), Inches(1.8), width=Inches(11.73), height=Inches(5.15))
    else:
        w_card9 = Inches(5.7)
        # Left: How XGBoost Works
        c_xgb1 = card(s9, Inches(0.8), Inches(1.8), w_card9, Inches(5.2), bg_col=WHITE)
        tf_xgb1 = c_xgb1.text_frame
        tf_xgb1.word_wrap = True
        tf_xgb1.margin_left = tf_xgb1.margin_right = tf_xgb1.margin_top = Inches(0.25)
        p = tf_xgb1.paragraphs[0]; p.text = "1. How XGBoost Works (The Core Mechanism)"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = FOREST
        for b_prefix, b_text in points_xgb_core:
            p = tf_xgb1.add_paragraph(); p.space_before = Pt(8)
            r1 = p.add_run(); r1.text = b_prefix + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = FOREST
            r2 = p.add_run(); r2.text = b_text; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

        # Right: Role in MaizeSight
        c_xgb2 = card(s9, Inches(6.8), Inches(1.8), w_card9, Inches(5.2), bg_col=WHITE, border_col=BORDER_GOLD)
        tf_xgb2 = c_xgb2.text_frame
        tf_xgb2.word_wrap = True
        tf_xgb2.margin_left = tf_xgb2.margin_right = tf_xgb2.margin_top = Inches(0.25)
        p = tf_xgb2.paragraphs[0]; p.text = "2. The Role of XGBoost in MaizeSight"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = GOLD
        for b_prefix, b_text in points_xgb_role:
            p = tf_xgb2.add_paragraph(); p.space_before = Pt(8)
            r1 = p.add_run(); r1.text = b_prefix + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = GOLD
            r2 = p.add_run(); r2.text = b_text; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

    add_notes(s9, "Slide 9 explains our primary tabular engine: XGBoost. Neural networks often struggle with heterogeneous tabular data, but XGBoost excels at fusing continuous embeddings, discrete canopy indices, and categorical hybrid parentages. Its sequential boosting with second-order Taylor expansion and L1/L2 penalties prevents overfitting on noisy agricultural plots, and its tree splits naturally model non-linear biological 'if-then' rules.")

    # ==========================================================
    # SLIDE 10: ALGORITHM SELECTION & IMPLEMENTATION
    # ==========================================================
    s10 = prs.slides.add_slide(blank_layout)
    bg(s10, LIGHT_BG)
    add_header(s10, "Model Architecture • Algorithm Selection", "Algorithm Selection & Architectural Ensembling", "Balancing deep visual representations with regularized, multi-algorithmic ensembles")

    cw10 = Inches(3.68)
    # Col 1: Vision Backbone
    c10_1 = card(s10, Inches(0.8), Inches(1.8), cw10, Inches(5.2), bg_col=WHITE)
    tf10_1 = c10_1.text_frame; tf10_1.word_wrap = True; tf10_1.margin_left = tf10_1.margin_right = tf10_1.margin_top = Inches(0.2)
    p = tf10_1.paragraphs[0]; p.text = "1. Vision Backbone: EfficientNet-B0"; p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = FOREST

    p10_1_items = [
        ("Why EfficientNet-B0?", "Ideal parameter efficiency (5.3M params vs 25M in ResNet-50) using Compound Scaling. Prevents overfitting on agricultural trial plots."),
        ("Pre-Training & Fine-Tuning:", "Pre-trained on ImageNet and fine-tuned end-to-end on multi-state maize plot patches."),
        ("1,280-D Representation:", "Final 1,000-class head truncated to output dense continuous feature representations for downstream ensembles.")
    ]
    for bp, bt in p10_1_items:
        p = tf10_1.add_paragraph(); p.space_before = Pt(10)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = FOREST
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

    # Col 2: Tabular Algorithms
    c10_2 = card(s10, Inches(4.82), Inches(1.8), cw10, Inches(5.2), bg_col=WHITE)
    tf10_2 = c10_2.text_frame; tf10_2.word_wrap = True; tf10_2.margin_left = tf10_2.margin_right = tf10_2.margin_top = Inches(0.2)
    p = tf10_2.paragraphs[0]; p.text = "2. Tabular Regressors"; p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = EMERALD

    p10_2_items = [
        ("XGBoost Regressor:", "Primary engine with low learning rates (eta = 0.018 - 0.03) and shallow tree depths (D in [2,4])."),
        ("ExtraTrees Regressor:", "Randomizes split thresholds across candidate features, suppressing variance from remote sensing noise."),
        ("HistGradientBoosting:", "Binned integer splits (256 bins) providing complementary decision boundaries."),
        ("RidgeCV (L2 Linear):", "Penalizes collinear canopy indices and establishes stable baseline predictions.")
    ]
    for bp, bt in p10_2_items:
        p = tf10_2.add_paragraph(); p.space_before = Pt(8)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = EMERALD
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

    # Col 3: Ensembling Architectures
    c10_3 = card(s10, Inches(8.84), Inches(1.8), cw10, Inches(5.2), bg_col=WHITE, border_col=BORDER_GOLD)
    tf10_3 = c10_3.text_frame; tf10_3.word_wrap = True; tf10_3.margin_left = tf10_3.margin_right = tf10_3.margin_top = Inches(0.2)
    p = tf10_3.paragraphs[0]; p.text = "3. Ensemble Architectures"; p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = GOLD

    p10_3_items = [
        ("Model 1 (Three-View Specialist):", "Blends Core View (113 features), Full View, and Physical Interaction View via learned meta-weights and gamma-variance calibration (R² = 0.8454)."),
        ("Model 2 (Decoupled Generalist):", "Stage 1: Field Physical Anchor (estimates field baseline).\nStage 2: Within-Field Ranker (sorts high vs low yielding hybrids).\nStage 3: 15-plot Bayesian/median calibration.")
    ]
    for bp, bt in p10_3_items:
        p = tf10_3.add_paragraph(); p.space_before = Pt(10)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = GOLD
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

    add_notes(s10, "Slide 10 details our architectural choices. Rather than relying on a single algorithm, we ensemble four distinct regressors: XGBoost, ExtraTrees for variance reduction, HistGradientBoosting for integer-binned decision trees, and Ridge regression for linear stability. In Model 1, three feature views are stacked together. In Model 2, we decouple field-level yield anchoring from within-field plot ranking.")

    # ==========================================================
    # SLIDE 11: TRAINING & VALIDATION METHODOLOGY
    # ==========================================================
    s11 = prs.slides.add_slide(blank_layout)
    bg(s11, LIGHT_BG)
    add_header(s11, "Experimental Methodology • Rigor & Validation", "Model Training & Validation Methodology", "Zero-leakage group splitting, dual validation schemes, and phased training workflow")

    train_diagram_path = "project__imagesss/training_pipeline_diagram.jpg"
    if os.path.exists(train_diagram_path):
        # Embed the generated infographic visual diagram directly
        s11.shapes.add_picture(train_diagram_path, Inches(0.8), Inches(1.8), width=Inches(11.73), height=Inches(5.15))
    else:
        w_card11 = Inches(5.7)

        # Left: Anti-Leakage Protocol & Phased Workflow
        c_m1 = card(s11, Inches(0.8), Inches(1.8), w_card11, Inches(5.2), bg_col=WHITE)
        tf_m1 = c_m1.text_frame; tf_m1.word_wrap = True; tf_m1.margin_left = tf_m1.margin_right = tf_m1.margin_top = Inches(0.2)
        p = tf_m1.paragraphs[0]; p.text = "1. Anti-Leakage Protocol & Phased Workflow"; p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = FOREST

        points_m1 = [
            ("The Multi-Temporal Leakage Danger:", "Each plot has 3 drone flights (TP1, TP2, TP3). Standard random splitting leaks temporal plot features into the test set."),
            ("Strict Plot-Level Group Splitting:", "All 3 flights for any plot were locked into the same split (Train, Val, or Test)."),
            ("Phase A (CNN Fine-Tuning):", "EfficientNet-B0 fine-tuned on 224x224 patches using AdamW with cosine learning rate decay."),
            ("Phase B (Frozen Feature State):", "Normalization scalers, prototype matrices, and kriging models fit only on the train split and frozen for inference."),
            ("Phase C & D (GBDT Tuning & Gamma Scaling):", "Grid search on tree depth and learning rate; post-hoc gamma calibration aligns ensemble prediction variance.")
        ]
        for bp, bt in points_m1:
            p = tf_m1.add_paragraph(); p.space_before = Pt(6)
            r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(9.5); r1.font.color.rgb = FOREST
            r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9); r2.font.color.rgb = SLATE_DARK

        # Right: Dual Validation Table & Metrics
        c_m2 = card(s11, Inches(6.8), Inches(1.8), w_card11, Inches(5.2), bg_col=WHITE, border_col=BORDER_EMERALD)
        tf_m2 = c_m2.text_frame; tf_m2.word_wrap = True; tf_m2.margin_left = tf_m2.margin_right = tf_m2.margin_top = Inches(0.2)
        p = tf_m2.paragraphs[0]; p.text = "2. Dual Validation Schemes & Evaluation Metrics"; p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = EMERALD

        # Add mini table
        t_m_shape = s11.shapes.add_table(5, 3, Inches(7.0), Inches(2.35), Inches(5.3), Inches(2.1))
        t_m = t_m_shape.table
        t_m.columns[0].width = Inches(1.7)
        t_m.columns[1].width = Inches(1.8)
        t_m.columns[2].width = Inches(1.8)

        val_headers = ["Dimension", "Model 1: Specialist", "Model 2: Generalist"]
        for i, h in enumerate(val_headers):
            cell = t_m.cell(0, i); cell.fill.solid(); cell.fill.fore_color.rgb = EMERALD
            p_c = cell.text_frame.paragraphs[0]; p_c.text = h; p_c.font.bold = True; p_c.font.size = Pt(8.5); p_c.font.color.rgb = WHITE

        val_rows = [
            ("Validation Scheme", "Stratified Group Split (5 sites)", "Leave-One-Location-Out (LOLO)"),
            ("Training Set", "1,489 plots (70%)", "1,313 plots (4 NE & IA sites)"),
            ("Validation Set", "319 plots (15%)", "330 plots (anchor/ranker tuning)"),
            ("Held-Out Test Set", "323 plots (15% across 5 sites)", "488 plots (100% unseen Crawfordsville)")
        ]
        for r_idx, r_vals in enumerate(val_rows, start=1):
            for c_idx, val in enumerate(r_vals):
                cell = t_m.cell(r_idx, c_idx); cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else RGBColor(245, 248, 246)
                p_c = cell.text_frame.paragraphs[0]; p_c.text = val; p_c.font.size = Pt(8); p_c.font.color.rgb = SLATE_DARK

        p_met = tf_m2.add_paragraph()
        p_met.space_before = Pt(175)
        p_met.text = "Evaluation Metrics Defined:"
        p_met.font.bold = True
        p_met.font.size = Pt(10)
        p_met.font.color.rgb = FOREST

        metric_defs = [
            ("R² (Variance Explained):", "1 - (SS_res / SS_tot). Evaluates rank and spread. Negative means worse than field mean."),
            ("RMSE (Root Mean Squared Error):", "In bu/ac, penalizing large prediction misses quadratically."),
            ("MAE (Mean Absolute Error):", "In bu/ac, intuitive real-world expected error for agronomists.")
        ]
        for bp, bt in metric_defs:
            p = tf_m2.add_paragraph(); p.space_before = Pt(3)
            r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(8.5); r1.font.color.rgb = FOREST
            r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(8); r2.font.color.rgb = SLATE_DARK

    add_notes(s11, "Slide 11 presents our validation rigor. Random splitting is disastrous with multi-temporal data because images of the same plot at different dates would leak across splits. We lock all three flights into the same partition. Furthermore, we test Model 2 using Leave-One-Location-Out: we train on Nebraska and Iowa, and evaluate purely on Crawfordsville, Indiana, which the model has never seen before.")

    # ==========================================================
    # SLIDE 12: RESULTS & PERFORMANCE ANALYSIS: MODEL 1 (Section v)
    # ==========================================================
    s12 = prs.slides.add_slide(blank_layout)
    bg(s12, LIGHT_BG)
    add_header(s12, "Section v • Results & Discussion", "Model 1 Results: Step-by-Step Performance Progression", "Ablation study on 323 held-out test plots across all 5 monitored research locations")

    # Left Container with Table & Findings
    w_left12 = Inches(6.8)
    t_shape12 = s12.shapes.add_table(6, 4, Inches(0.8), Inches(1.8), w_left12, Inches(2.2))
    t12 = t_shape12.table
    t12.columns[0].width = Inches(2.5)
    t12.columns[1].width = Inches(1.1)
    t12.columns[2].width = Inches(1.4)
    t12.columns[3].width = Inches(1.8)

    m1_headers = ["Progression Stage", "R² Score", "RMSE (bu/ac)", "Performance Gain"]
    for i, h in enumerate(m1_headers):
        cell = t12.cell(0, i); cell.fill.solid(); cell.fill.fore_color.rgb = FOREST
        p_c = cell.text_frame.paragraphs[0]; p_c.text = h; p_c.font.bold = True; p_c.font.size = Pt(9); p_c.font.color.rgb = WHITE

    m1_rows = [
        ("Step 1: Raw TP3 Vision Model", "0.7940", "24.11", "Single-flight baseline"),
        ("Step 2: Multiple Flights (TP1–TP3)", "0.8087", "23.23", "+1.47% R² via growth dynamics"),
        ("Step 3: Vision + Canopy Indices", "0.8178", "22.67", "-0.56 bu/ac RMSE via color math"),
        ("Step 4: Pedigree + Spatial Kriging", "0.8436", "21.01", "+2.58% R² via micro-soil mapping"),
        ("Step 5: 3-View Stacked Ensemble", "0.8454", "20.88", "Final specialist precision")
    ]
    for r_idx, r_vals in enumerate(m1_rows, start=1):
        for c_idx, val in enumerate(r_vals):
            cell = t12.cell(r_idx, c_idx); cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else RGBColor(245, 248, 246)
            p_c = cell.text_frame.paragraphs[0]; p_c.text = val; p_c.font.size = Pt(8.5); p_c.font.color.rgb = SLATE_DARK
            if c_idx == 1:
                p_c.font.bold = True
                p_c.font.color.rgb = EMERALD if r_idx == 5 else FOREST

    # Bottom Analysis Container under Table
    c_m1_bot = card(s12, Inches(0.8), Inches(4.2), w_left12, Inches(2.8), bg_col=WHITE, border_col=BORDER_EMERALD)
    tf_bot = c_m1_bot.text_frame; tf_bot.word_wrap = True; tf_bot.margin_left = tf_bot.margin_right = tf_bot.margin_top = Inches(0.2)
    p = tf_bot.paragraphs[0]; p.text = "Key Analytical Findings for Model 1 (Specialist)"; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = FOREST

    points_m1_bot = [
        ("Multi-Temporal Triad Advantage:", "Moving from a single-flight flowering snapshot to multi-temporal tracking lowered RMSE by nearly 0.9 bu/ac, proving stand count and vegetative rate add essential signal."),
        ("Canopy Indices Complement Vision:", "Adding 12 explicit canopy ratios (ExG, VARI, FGCC) further reduced RMSE by 0.56 bu/ac via physiological color constraints."),
        ("Spatial Kriging & Genetic Lineage:", "Step 4 produced the largest single leap, dropping RMSE by over 1.66 bu/ac by modeling hybrid heterosis and micro-soil trends."),
        ("Final Stacked Precision:", "The 3-view stacked ensemble with gamma calibration achieved R² = 0.8454 and MAE = 16.53 bu/ac, explaining 84.5% of yield variance.")
    ]
    for bp, bt in points_m1_bot:
        p = tf_bot.add_paragraph(); p.space_before = Pt(3)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(8.5); r1.font.color.rgb = FOREST
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(8); r2.font.color.rgb = SLATE_DARK

    # Right Container with Comparison Chart Image
    chart_path12 = "project__imagesss/step_comparison_charts/1_R2_Comparison_Model1_vs_Model2.png"
    if os.path.exists(chart_path12):
        s12.shapes.add_picture(chart_path12, Inches(7.8), Inches(1.8), width=Inches(4.7), height=Inches(5.2))

    add_notes(s12, "Slide 12 presents the ablation results for Model 1 across 323 held-out test plots. A raw single-flight CNN achieves an R-squared of 0.794. Adding multi-temporal dynamics boosts it to 0.809. Fusing canopy indices pushes it to 0.818, and incorporating spatial kriging and hybrid genetics drops RMSE by over 1.6 bushels per acre. Our final three-view ensemble reaches an R-squared of 0.8454 with an average error of only 16.5 bu/ac.")

    # ==========================================================
    # SLIDE 13: RESULTS & PERFORMANCE ANALYSIS: MODEL 2 (Section v)
    # ==========================================================
    s13 = prs.slides.add_slide(blank_layout)
    bg(s13, LIGHT_BG)
    add_header(s13, "Section v • Results & Discussion", "Model 2 Results: Overcoming the Cross-Site Domain Shift", "Leave-One-Location-Out evaluation on 488 unseen test plots at Crawfordsville, Indiana")

    # Left Container with Table & Findings
    w_left13 = Inches(6.8)
    t_shape13 = s13.shapes.add_table(7, 4, Inches(0.8), Inches(1.8), w_left13, Inches(2.4))
    t13 = t_shape13.table
    t13.columns[0].width = Inches(2.5)
    t13.columns[1].width = Inches(1.1)
    t13.columns[2].width = Inches(1.4)
    t13.columns[3].width = Inches(1.8)

    m2_headers = ["Progression Stage", "R² Score", "RMSE (bu/ac)", "Performance Discussion"]
    for i, h in enumerate(m2_headers):
        cell = t13.cell(0, i); cell.fill.solid(); cell.fill.fore_color.rgb = FOREST
        p_c = cell.text_frame.paragraphs[0]; p_c.text = h; p_c.font.bold = True; p_c.font.size = Pt(9); p_c.font.color.rgb = WHITE

    m2_rows = [
        ("Step 1: Frozen TP3 Vision Backbone", "-1.6207", "39.95", "Severe failure of naive transfer"),
        ("Step 2: Multi-Flight + Planting Date", "-0.6072", "31.29", "Environmental offset correction"),
        ("Step 3: Fine-Tuned Backbone", "-0.4431", "29.65", "Feature adaptation to maize"),
        ("Step 4: LOLO Normalization + Spatial", "-0.2514", "27.61", "Mitigating site-specific bias"),
        ("Step 5: Final Zero-Shot Ensemble", "+0.1398", "22.89", "Flips negative R² to positive"),
        ("Step 6: With 15-Plot Calibration", "+0.2168", "21.89", "Rapid adaptation with minimal labels")
    ]
    for r_idx, r_vals in enumerate(m2_rows, start=1):
        for c_idx, val in enumerate(r_vals):
            cell = t13.cell(r_idx, c_idx); cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else RGBColor(245, 248, 246)
            p_c = cell.text_frame.paragraphs[0]; p_c.text = val; p_c.font.size = Pt(8.5); p_c.font.color.rgb = SLATE_DARK
            if c_idx == 1:
                p_c.font.bold = True
                p_c.font.color.rgb = CORAL if "-" in val else EMERALD

    # Bottom Analysis Container under Table
    c_m2_bot = card(s13, Inches(0.8), Inches(4.35), w_left13, Inches(2.65), bg_col=WHITE, border_col=BORDER_GOLD)
    tf_bot2 = c_m2_bot.text_frame; tf_bot2.word_wrap = True; tf_bot2.margin_left = tf_bot2.margin_right = tf_bot2.margin_top = Inches(0.2)
    p = tf_bot2.paragraphs[0]; p.text = "Key Analytical Findings for Model 2 (Generalist)"; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = GOLD

    points_m2_bot = [
        ("Failure of Naive Transfer:", "Off-the-shelf CNN transfer failed catastrophically at Crawfordsville with R² = -1.62 due to severe environmental baseline yield differences between Nebraska and Indiana."),
        ("Decoupled Anchor-Ranker:", "By decoupling the prediction into a macro Field Anchor (field baseline) and a Within-Field Ranker (relative plot vigor), Step 5 flipped R² to positive (+0.1398) with zero training data on the site."),
        ("Rapid Few-Shot Field Calibration:", "Supplying just 15 labeled reference plots (Step 6) boosted test R² to +0.2168 and dropped RMSE to 21.89 bu/ac, demonstrating high commercial utility.")
    ]
    for bp, bt in points_m2_bot:
        p = tf_bot2.add_paragraph(); p.space_before = Pt(3)
        r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(8.5); r1.font.color.rgb = FOREST
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(8); r2.font.color.rgb = SLATE_DARK

    # Right Container with Crawfordsville Test Chart
    chart_path13 = "project__imagesss/crawfordsville_test_comparison_chart.png"
    if os.path.exists(chart_path13):
        s13.shapes.add_picture(chart_path13, Inches(7.8), Inches(1.8), width=Inches(4.7), height=Inches(5.2))

    add_notes(s13, "Slide 13 highlights our breakthrough in out-of-domain transfer learning. Naive transfer learning scored negative 1.62 at Crawfordsville, Indiana. Our decoupled Anchor-Ranker architecture broke through this barrier, achieving a positive R-squared of +0.140 in pure zero-shot mode. When just 15 reference plots were provided for calibration, R-squared reached +0.217 with an RMSE of 21.89 bu/ac.")

    # ==========================================================
    # SLIDE 14: COMPARISON OF ALTERNATIVE ALGORITHMS & APPROACHES
    # ==========================================================
    s14 = prs.slides.add_slide(blank_layout)
    bg(s14, LIGHT_BG)
    add_header(s14, "Algorithmic Benchmarking • Architectural Decisions", "Comparison of Alternative Algorithms & Approaches", "Why conventional deep learning and single-flight setups fell short")

    # Benchmark Matrix Table
    t_shape14 = s14.shapes.add_table(6, 4, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    t14 = t_shape14.table
    t14.columns[0].width = Inches(2.2)
    t14.columns[1].width = Inches(2.9)
    t14.columns[2].width = Inches(2.9)
    t14.columns[3].width = Inches(3.7)

    comp_headers = ["Design Dimension", "Conventional / Alternative Approach", "MaizeSight Chosen Approach", "Key Advantage / Empirical Outcome"]
    for i, h in enumerate(comp_headers):
        cell = t14.cell(0, i); cell.fill.solid(); cell.fill.fore_color.rgb = FOREST
        p_c = cell.text_frame.paragraphs[0]; p_c.text = h; p_c.font.bold = True; p_c.font.size = Pt(9.5); p_c.font.color.rgb = WHITE

    comp_rows = [
        ("Vision Backbone", "ResNet-50 / VGG-16 / ViT\n(25M–86M parameters; overfits on trial plots)", "EfficientNet-B0 (Compound Scaled)\n(5.3M parameters; MBConv + SE attention)", "Lower parameter footprint avoids overfitting; 4x faster feature extraction throughput"),
        ("Plot Tiling & Edge", "Zero-Padding (cv2.BORDER_CONSTANT)\n(Creates artificial high-contrast black edges)", "Replication Padding (BORDER_REPLICATE)\n(Extends edge pixels outward smoothly)", "Preserves perimeter leaf geometry and row margins; eliminates fake zero-gradient edges"),
        ("Sensing Strategy", "Single-Flight Snapshot (Flowering only)\n(Evaluates canopy solely at anthesis / R1)", "Multi-Temporal Triad (TP1 + TP2 + TP3)\n(Emergence, canopy closure, and flowering)", "Captures growth velocity deltas; yields +1.5% R² gain over single-flight models"),
        ("Tabular Predictor", "Single Standalone XGBoost / Random Forest\n(Prone to algorithm-specific variance)", "Multi-View Stacked Ensemble\n(XGBoost + ExtraTrees + HistGB + Ridge)", "Algorithmic diversity stabilizes predictions; pushes in-domain R² to 0.845"),
        ("Cross-Site Transfer", "Direct Monolithic Regression\n(Scored negative R² = -1.62 on unseen Indiana site)", "Decoupled Anchor + Ranker\n(Estimates field baseline separately from rank)", "Flips negative R² to positive (+0.140 zero-shot, +0.217 with 15 calibration plots)")
    ]
    for r_idx, r_vals in enumerate(comp_rows, start=1):
        for c_idx, val in enumerate(r_vals):
            cell = t14.cell(r_idx, c_idx); cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else RGBColor(245, 248, 246)
            p_c = cell.text_frame.paragraphs[0]; p_c.text = val; p_c.font.size = Pt(8.5); p_c.font.color.rgb = SLATE_DARK
            if c_idx == 0:
                p_c.font.bold = True
                p_c.font.color.rgb = FOREST
            elif c_idx == 2:
                p_c.font.bold = True
                p_c.font.color.rgb = EMERALD

    add_notes(s14, "Slide 14 provides a comprehensive benchmark comparison. We show why heavy backbones like ResNet-50 overfit on 2,000 plots compared to EfficientNet-B0; why zero padding corrupts edge features compared to replication padding; why single-flight sensing hits an R-squared ceiling of 0.794; and why monolithic transfer fails catastrophically compared to our decoupled Anchor-Ranker approach.")

    # ==========================================================
    # SLIDE 15: SOFTWARE IMPLEMENTATION (FastAPI + Three.js)
    # ==========================================================
    s15 = prs.slides.add_slide(blank_layout)
    bg(s15, LIGHT_BG)
    add_header(s15, "System Implementation • Production Deployment", "Software Implementation: The MaizeSight Web Platform", "FastAPI asynchronous backend, 3D geospatial visualization, and real-time dual-model inference")

    infer_diagram_path = "project__imagesss/inference_pipeline_diagram.jpg"
    if os.path.exists(infer_diagram_path):
        # Embed the generated infographic visual diagram directly
        s15.shapes.add_picture(infer_diagram_path, Inches(0.8), Inches(1.8), width=Inches(11.73), height=Inches(5.15))
    else:
        w_card15 = Inches(5.7)

        # Left: Architecture & Tech Stack
        c_s1 = card(s15, Inches(0.8), Inches(1.8), w_card15, Inches(5.2), bg_col=WHITE)
        tf_s1 = c_s1.text_frame; tf_s1.word_wrap = True; tf_s1.margin_left = tf_s1.margin_right = tf_s1.margin_top = Inches(0.25)
        p = tf_s1.paragraphs[0]; p.text = "1. System Architecture & Tech Stack"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = FOREST

        points_s1 = [
            ("High-Performance Backend (FastAPI / Python 3.13):", "Asynchronous RESTful API exposing live endpoints: `/predict_both`, `/api/database/plots`, `/api/locations`. Lazy-loaded pipeline manages model bundles and CUDA/CPU device allocation."),
            ("Modern 3D Frontend (Three.js + GSAP + ES6):", "• 3D Geospatial Canvas: Interactive tiltable topographic map of the US Corn Belt rendering trial sites across Nebraska, Iowa, and Indiana.\n• Scroll-Driven Growth Animation: 3D maize plant evolves dynamically from germination to flowering."),
            ("One-Click Local & Server Deployment:", "Packaged with automated startup scripts (`run_app.bat` / `run_app.ps1`) for instant startup and auto-launching the default browser.")
        ]
        for bp, bt in points_s1:
            p = tf_s1.add_paragraph(); p.space_before = Pt(10)
            r1 = p.add_run(); r1.text = bp + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = FOREST
            r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

        # Right: Core Modules
        c_s2 = card(s15, Inches(6.8), Inches(1.8), w_card15, Inches(5.2), bg_col=WHITE, border_col=BORDER_EMERALD)
        tf_s2 = c_s2.text_frame; tf_s2.word_wrap = True; tf_s2.margin_left = tf_s2.margin_right = tf_s2.margin_top = Inches(0.25)
        p = tf_s2.paragraphs[0]; p.text = "2. Core Functional Modules & Working Capabilities"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = EMERALD

        points_s2 = [
            ("A. Automated Multi-Temporal Regex Matcher:", "Built-in regex engine (`Location-TP-Exp_Range_Row.PNG`) automatically detects and binds TP1, TP2, and TP3 drone images to GroundTruth plot records without manual entry."),
            ("B. Dual-Model Side-by-Side Inference (`/predict_both`):", "Runs Model 1 and Model 2 simultaneously, reporting estimated yield (bu/ac), within-field percentile rank, and model agreement metrics."),
            ("C. Automated Diagnostic Quality Flagging:", "Issues data quality alerts:\n• `tp3_image_missing`: Critical flowering flight absent.\n• `image_error_corrupt`: Flags unreadable image tiles.\n• `small_field_warning`: Notifies if incoming batch has <30 plots."),
            ("D. In-Memory Database Explorer:", "Indexes all 2,131 verified research plots in memory for instantaneous sub-millisecond filtering and search.")
        ]
        for bp, bt in points_s2:
            p = tf_s2.add_paragraph(); p.space_before = Pt(8)
            r1 = p.add_run(); r1.text = bp + " "; r1.font.bold = True; r1.font.size = Pt(10); r1.font.color.rgb = EMERALD
            r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9.5); r2.font.color.rgb = SLATE_DARK

    add_notes(s15, "Slide 15 showcases our production software implementation. We built a full-stack platform called MaizeSight. The FastAPI backend runs dual-model inference in milliseconds, while the frontend provides an interactive Three.js 3D map, automated file matching that links drone flight PNGs by plot ID, and diagnostic flags that protect agronomists from corrupt or missing image inputs.")

    # ==========================================================
    # SLIDE 16: CONCLUSION & SCIENTIFIC LEARNINGS (Section vi)
    # ==========================================================
    s16 = prs.slides.add_slide(blank_layout)
    bg(s16, LIGHT_BG)
    add_header(s16, "Section vi • Conclusion & Takeaways", "Conclusion: Key Insights & Scientific Learnings", "Summary of project contributions and fundamental takeaways for agricultural AI")

    cw16 = Inches(5.7)
    ch16 = Inches(2.45)

    c16_1 = card(s16, Inches(0.8), Inches(1.8), cw16, ch16, bg_col=WHITE)
    tf16_1 = c16_1.text_frame; tf16_1.word_wrap = True; tf16_1.margin_left = tf16_1.margin_right = tf16_1.margin_top = Inches(0.2)
    p = tf16_1.paragraphs[0]; p.text = "1. Multi-Temporal Sensing is Non-Negotiable"; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = FOREST
    p_b = tf16_1.add_paragraph(); p_b.space_before = Pt(4)
    p_b.text = "Single-flight snapshots cannot separate temporary weather stress from true physiological potential. Combining early emergence (TP1), peak canopy closure (TP2), and flowering anthesis (TP3) is essential for high-confidence harvest estimation."
    p_b.font.size = Pt(9.5); p_b.font.color.rgb = SLATE_DARK

    c16_2 = card(s16, Inches(6.8), Inches(1.8), cw16, ch16, bg_col=WHITE)
    tf16_2 = c16_2.text_frame; tf16_2.word_wrap = True; tf16_2.margin_left = tf16_2.margin_right = tf16_2.margin_top = Inches(0.2)
    p = tf16_2.paragraphs[0]; p.text = "2. Hybrid Feature Fusion Trumps Pure Vision"; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = EMERALD
    p_b = tf16_2.add_paragraph(); p_b.space_before = Pt(4)
    p_b.text = "Coupling deep CNN embeddings (EfficientNet-B0) with biophysical canopy indices (ExG, VARI), hybrid parental pedigrees, and 2D spatial kriging significantly outperforms black-box image-to-yield neural networks."
    p_b.font.size = Pt(9.5); p_b.font.color.rgb = SLATE_DARK

    c16_3 = card(s16, Inches(0.8), Inches(4.5), cw16, ch16, bg_col=WHITE)
    tf16_3 = c16_3.text_frame; tf16_3.word_wrap = True; tf16_3.margin_left = tf16_3.margin_right = tf16_3.margin_top = Inches(0.2)
    p = tf16_3.paragraphs[0]; p.text = "3. Decoupling Solves the Cross-Site Domain Shift"; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = GOLD
    p_b = tf16_3.add_paragraph(); p_b.space_before = Pt(4)
    p_b.text = "Monolithic models fail when deployed to new farms because baseline yields vary wildly. Decoupling macro field-level yield anchoring from within-field spatial ranking flips negative cross-site R² to positive zero-shot performance."
    p_b.font.size = Pt(9.5); p_b.font.color.rgb = SLATE_DARK

    c16_4 = card(s16, Inches(6.8), Inches(4.5), cw16, ch16, bg_col=WHITE)
    tf16_4 = c16_4.text_frame; tf16_4.word_wrap = True; tf16_4.margin_left = tf16_4.margin_right = tf16_4.margin_top = Inches(0.2)
    p = tf16_4.paragraphs[0]; p.text = "4. Few-Shot Field Calibration is Practical & Scalable"; p.font.bold = True; p.font.size = Pt(11.5); p.font.color.rgb = FOREST
    p_b = tf16_4.add_paragraph(); p_b.space_before = Pt(4)
    p_b.text = "Measuring as few as 15 reference calibration plots in an unfamiliar field boosts generalization R² to +0.217 and drops RMSE to 21.89 bu/ac, providing a highly cost-effective operational blueprint for commercial growers."
    p_b.font.size = Pt(9.5); p_b.font.color.rgb = SLATE_DARK

    add_notes(s16, "Slide 16 concludes our core findings. Multi-temporal sensing is non-negotiable; hybrid feature fusion beats pure pixel deep learning; decoupling field anchors from within-field rankers solves the cross-site transfer barrier; and measuring just 15 reference plots makes few-shot adaptation practical in real-world agriculture.")

    # ==========================================================
    # SLIDE 17: FUTURE ENHANCEMENTS & ROADMAP (Section vi)
    # ==========================================================
    s17 = prs.slides.add_slide(blank_layout)
    bg(s17, LIGHT_BG)
    add_header(s17, "Section vi • Future Enhancements", "Future Enhancements & Research Roadmap", "Scaling MaizeSight from research trial plots to continental production agriculture")

    cw17 = Inches(2.72)
    t17 = Inches(1.8)
    h17 = Inches(5.2)

    pillars = [
        ("1. Multispectral & Thermal Fusion", FOREST, [
            ("Thermal Infrared:", "Capture canopy surface temperature and transpiration cooling to detect early moisture stress."),
            ("RedEdge Wavebands:", "Directly quantify chlorophyll-a/b absorption for precise nitrogen status mapping.")
        ]),
        ("2. Satellite-UAV Harmonization", EMERALD, [
            ("Harmonized Constellations:", "Fuse high-frequency UAV imagery with Sentinel-2 and PlanetScope satellite data."),
            ("Gap-Filling:", "Use satellite temporal curves to bridge cloudy periods between scheduled drone flights.")
        ]),
        ("3. Spatio-Temporal Transformers", GOLD, [
            ("Vision Transformers (ViT):", "Replace manual multi-step pooling with 3D spatio-temporal self-attention."),
            ("Dynamic Phenology:", "Learn cross-flight attention weights to track variable crop emergence timelines.")
        ]),
        ("4. Edge AI On-Drone Inference", FOREST, [
            ("Quantization & TensorRT:", "Convert PyTorch / XGBoost models to INT8 ONNX/TensorRT engines."),
            ("NVIDIA Jetson Payloads:", "Run live yield estimation and diagnostic flagging on companion drone computers in flight.")
        ])
    ]

    for idx, (title, color, items) in enumerate(pillars):
        left_pos = Inches(0.8 + idx * 2.98)
        c = card(s17, left_pos, t17, cw17, h17, bg_col=WHITE, border_col=color if idx == 1 or idx == 2 else BORDER_LIGHT)
        tf = c.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.2)
        p = tf.paragraphs[0]; p.text = title; p.font.bold = True; p.font.size = Pt(11); p.font.color.rgb = color
        for bp, bt in items:
            p = tf.add_paragraph(); p.space_before = Pt(12)
            r1 = p.add_run(); r1.text = "• " + bp + " "; r1.font.bold = True; r1.font.size = Pt(9.5); r1.font.color.rgb = color
            r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9); r2.font.color.rgb = SLATE_DARK

    add_notes(s17, "Slide 17 looks forward. We outline four high-impact directions: fusing multispectral and thermal sensors for water stress detection; combining satellite and drone data for county-wide coverage; implementing spatio-temporal Transformers; and quantizing our models for live edge inference on NVIDIA Jetson computers mounted directly on drones.")

    # ==========================================================
    # SLIDE 18: REFERENCES (Section vii)
    # ==========================================================
    s18 = prs.slides.add_slide(blank_layout)
    bg(s18, LIGHT_BG)
    add_header(s18, "Section vii • Academic References", "References & Foundational Literature", "Peer-reviewed literature across remote sensing, deep vision, gradient boosting, and agronomy")

    w_card18 = Inches(5.7)

    # Left Column References
    c_r1 = card(s18, Inches(0.8), Inches(1.8), w_card18, Inches(5.2), bg_col=WHITE)
    tf_r1 = c_r1.text_frame; tf_r1.word_wrap = True; tf_r1.margin_left = tf_r1.margin_right = tf_r1.margin_top = Inches(0.2)
    p = tf_r1.paragraphs[0]; p.text = "Remote Sensing & Agricultural Data"; p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = FOREST

    refs_left = [
        ("1. Shrestha, S., et al. (2022).", "Crop performance, aerial, and satellite data from multistate maize yield trials. Dryad Digital Repository. doi:10.5061/dryad.905qftttm."),
        ("2. Woebbecke, D. M., et al. (1995).", "Color indices for weed identification under various soil, residue, and lighting conditions. Transactions of the ASAE, 38(1), 259–269. (Foundation for Excess Green - ExG)."),
        ("3. Gitelson, A. A., et al. (2002).", "Novel algorithms for remote estimation of vegetation fraction. Remote Sensing of Environment, 80(1), 76–87. (Visible Atmospherically Resistant Index - VARI).")
    ]
    for bp, bt in refs_left:
        p = tf_r1.add_paragraph(); p.space_before = Pt(12)
        r1 = p.add_run(); r1.text = bp + "\n"; r1.font.bold = True; r1.font.size = Pt(9.5); r1.font.color.rgb = FOREST
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9); r2.font.color.rgb = SLATE_DARK

    # Right Column References
    c_r2 = card(s18, Inches(6.8), Inches(1.8), w_card18, Inches(5.2), bg_col=WHITE)
    tf_r2 = c_r2.text_frame; tf_r2.word_wrap = True; tf_r2.margin_left = tf_r2.margin_right = tf_r2.margin_top = Inches(0.2)
    p = tf_r2.paragraphs[0]; p.text = "Computer Vision & Machine Learning"; p.font.bold = True; p.font.size = Pt(12); p.font.color.rgb = EMERALD

    refs_right = [
        ("4. Tan, M., & Le, Q. V. (2019).", "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. International Conference on Machine Learning (ICML), PMLR, 6105–6114."),
        ("5. Chen, T., & Guestrin, C. (2016).", "XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794."),
        ("6. Peng, H., Long, F., & Ding, C. (2005).", "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. IEEE TPAMI, 27(8), 1226–1238.")
    ]
    for bp, bt in refs_right:
        p = tf_r2.add_paragraph(); p.space_before = Pt(12)
        r1 = p.add_run(); r1.text = bp + "\n"; r1.font.bold = True; r1.font.size = Pt(9.5); r1.font.color.rgb = EMERALD
        r2 = p.add_run(); r2.text = bt; r2.font.size = Pt(9); r2.font.color.rgb = SLATE_DARK

    add_notes(s18, "Slide 18 provides formal citations for our work, recognizing Shrestha et al.'s open dataset on Dryad, classical remote sensing index papers by Woebbecke and Gitelson, and fundamental machine learning architectures including EfficientNet and XGBoost.")

    # ==========================================================
    # SLIDE 19: CONTRIBUTORS & ACKNOWLEDGEMENTS (Section viii)
    # ==========================================================
    s19 = prs.slides.add_slide(blank_layout)
    bg(s19, DARK_BG)

    # Top accent line
    bar19 = s19.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
    bar19.fill.solid(); bar19.fill.fore_color.rgb = GOLD; bar19.line.fill.background()

    add_header(s19, "Section viii • Project Team & Acknowledgements", "Contributors & Acknowledgements", "Multidisciplinary collaboration spanning machine learning, remote sensing, agronomy, and software engineering", is_dark=True)

    cw19 = Inches(2.72)
    t19 = Inches(1.8)
    h19 = Inches(3.6)

    team_members = [
        ("Machine Learning Lead", "EfficientNet-B0 Backbone", "Fine-tuned deep vision models, extracted 1,280-D patch embeddings, and engineered Model 1 three-view stacked ensemble."),
        ("Data & Remote Sensing", "UAV Image Preprocessing", "Audited 7,281 drone files, implemented replication padding, extracted 12 biophysical canopy indices, and computed temporal velocities."),
        ("Cross-Domain Lead", "Decoupled Architecture", "Designed Leave-One-Location-Out cross-validation, formulated Anchor-Ranker architecture, and developed few-shot calibration."),
        ("Full-Stack Architect", "FastAPI & 3D Web Platform", "Constructed high-speed asynchronous REST API, interactive Three.js 3D terrain canvas, automated regex file matcher, and quality flaggers.")
    ]

    for idx, (role, focus, desc) in enumerate(team_members):
        left_pos = Inches(0.8 + idx * 2.98)
        c = card(s19, left_pos, t19, cw19, h19, bg_col=DARK_CARD, border_col=DARK_CARD_BORDER)
        tf = c.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.2)
        p = tf.paragraphs[0]; p.text = f"Contributor {idx+1}"; p.font.bold = True; p.font.size = Pt(13); p.font.color.rgb = GOLD
        p2 = tf.add_paragraph(); p2.text = role; p2.font.bold = True; p2.font.size = Pt(10); p2.font.color.rgb = WHITE; p2.space_before = Pt(3)
        p3 = tf.add_paragraph(); p3.text = focus; p3.font.size = Pt(9); p3.font.color.rgb = RGBColor(165, 214, 167); p3.space_before = Pt(2)
        p4 = tf.add_paragraph(); p4.text = desc; p4.font.size = Pt(8.5); p4.font.color.rgb = RGBColor(190, 215, 205); p4.space_before = Pt(8)

    # Bottom Acknowledgement & Q&A Banner
    banner = card(s19, Inches(0.8), Inches(5.65), Inches(11.7), Inches(1.35), bg_col=DARK_CARD, border_col=GOLD, b_width=1.5)
    tf_ban = banner.text_frame; tf_ban.word_wrap = True; tf_ban.margin_left = tf_ban.margin_right = tf_ban.margin_top = Inches(0.18)
    p_b1 = tf_ban.paragraphs[0]; p_b1.text = "PROJECT ACKNOWLEDGEMENTS & OPEN SOURCE ECOSYSTEM"; p_b1.font.bold = True; p_b1.font.size = Pt(10.5); p_b1.font.color.rgb = GOLD
    p_b2 = tf_ban.add_paragraph()
    p_b2.text = "Special thanks to project mentors, research faculty, and the open-source community powering PyTorch, XGBoost, Scikit-learn, OpenCV, FastAPI, and Three.js.\nThank you for your attention! We now welcome questions and discussion."
    p_b2.font.size = Pt(9.5); p_b2.font.color.rgb = WHITE; p_b2.space_before = Pt(4)

    add_notes(s19, "Thank you everyone for your attention. We would like to acknowledge our project advisors, the authors of the open Dryad dataset, and the open-source software libraries that made this system possible. We are now open for any questions!")

    # Save presentation
    output_filename = "MaizeSight_CornBelt_Yield_Presentation.pptx"
    prs.save(output_filename)
    print(f"Presentation saved successfully to {output_filename} with {len(prs.slides)} slides!")

if __name__ == "__main__":
    create_deck()
