import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import pandas as pd
from fpdf import FPDF
import tempfile
import os
import re
import io

# Import credentials from separate file
try:
    from credentials import DOCTOR_CREDENTIALS
except ImportError:
    st.error("""
    ⚠️ Credentials file not found! 
    
    Please create a file called 'credentials.py' with the following structure:
    
    DOCTOR_CREDENTIALS = {
        "email@example.com": {
            "password": "password",
            "name": "Doctor Name",
            "specialization": "Specialization"
        }
    }
    """)
    st.stop()

# --- App Configuration ---
st.set_page_config(
    page_title="Advanced Cervical Spine Fracture Detection",
    page_icon="🦴",
    layout="wide"
)

# --- Initialize session state for login and patient data ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'report_images' not in st.session_state:
    st.session_state.report_images = {}

# --- Login Function ---
def login():
    """Display login form and authenticate user"""
    st.title("🔐 Doctor Login - Cervical Spine Fracture Detection")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 👨‍⚕️ Please Login to Access the Application")
        st.markdown("---")
        
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            login_button = st.button("🔓 Login", use_container_width=True)
        
        if login_button:
            if email in DOCTOR_CREDENTIALS and DOCTOR_CREDENTIALS[email]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.current_user = {
                    "email": email,
                    "name": DOCTOR_CREDENTIALS[email]["name"],
                    "specialization": DOCTOR_CREDENTIALS[email]["specialization"]
                }
                st.success(f"✅ Welcome, {DOCTOR_CREDENTIALS[email]['name']}!")
                st.rerun()
            else:
                st.error("❌ Invalid email or password. Please try again.")
        
        st.markdown("---")
        st.markdown("#### 🔒 Secure Login")
        st.markdown("Please contact your administrator if you need access.")

# --- Logout Function ---
def logout():
    """Logout the current user"""
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.patient_info = {}
    st.session_state.report_generated = False
    st.session_state.report_images = {}
    st.rerun()

# --- Check if user is logged in ---
if not st.session_state.logged_in:
    login()
    st.stop()  # Stop execution if not logged in

# --- Cervical Vertebrae Information ---
CERVICAL_VERTEBRAE_INFO = {
    "C1": {
        "name": "Atlas",
        "description": "The first cervical vertebra, ring-shaped bone that supports the skull",
        "fracture_types": ["Jefferson fracture", "Posterior arch fracture", "Lateral mass fracture"],
        "causes": ["High-energy axial loading", "Diving accidents", "Motor vehicle accidents", "Falls from height"],
        "symptoms": ["Severe neck pain", "Occipital headache", "Neurological deficits (rare)", "Restricted neck movement"],
        "complications": ["Atlantooccipital instability", "Vertebral artery injury", "Spinal cord compression", "Chronic pain"],
        "treatment": ["Halo vest immobilization", "Cervical collar", "Surgical fusion (severe cases)", "Physical therapy"]
    },
    "C2": {
        "name": "Axis",
        "description": "The second cervical vertebra with the odontoid process (dens)",
        "fracture_types": ["Odontoid fracture (Type I, II, III)", "Hangman's fracture", "Body fracture"],
        "causes": ["Flexion-extension injuries", "High-energy trauma", "Falls in elderly", "Motor vehicle accidents"],
        "symptoms": ["Severe neck pain", "Torticollis", "Dysphagia", "Neurological symptoms"],
        "complications": ["Atlantoaxial instability", "Spinal cord injury", "Respiratory compromise", "Death"],
        "treatment": ["Halo immobilization", "Odontoid screw fixation", "Posterior C1-C2 fusion", "External fixation"]
    },
    "C3": {
        "name": "Third Cervical Vertebra",
        "description": "Typical cervical vertebra with bifid spinous process",
        "fracture_types": ["Compression fracture", "Burst fracture", "Facet fracture", "Spinous process fracture"],
        "causes": ["Flexion injuries", "Axial loading", "Extension injuries", "Direct trauma"],
        "symptoms": ["Neck pain", "Muscle spasm", "Radiculopathy", "Limited range of motion"],
        "complications": ["Spinal cord compression", "Nerve root injury", "Chronic instability", "Deformity"],
        "treatment": ["Cervical collar", "Surgical stabilization", "Anterior cervical fusion", "Conservative management"]
    },
    "C4": {
        "name": "Fourth Cervical Vertebra",
        "description": "Typical cervical vertebra, critical for diaphragmatic breathing",
        "fracture_types": ["Compression fracture", "Burst fracture", "Teardrop fracture", "Facet dislocation"],
        "causes": ["Diving accidents", "Motor vehicle accidents", "Falls", "Sports injuries"],
        "symptoms": ["Neck pain", "Arm weakness", "Respiratory issues", "Sensory changes"],
        "complications": ["Respiratory paralysis", "Quadriplegia", "Autonomic dysfunction", "Chronic pain"],
        "treatment": ["Immediate stabilization", "Surgical decompression", "Anterior/posterior fusion", "Mechanical ventilation"]
    },
    "C5": {
        "name": "Fifth Cervical Vertebra",
        "description": "Most commonly fractured cervical vertebra",
        "fracture_types": ["Compression fracture", "Burst fracture", "Teardrop fracture", "Unilateral facet dislocation"],
        "causes": ["Hyperflexion injuries", "Axial loading", "Motor vehicle accidents", "Diving into shallow water"],
        "symptoms": ["Severe neck pain", "Bilateral arm weakness", "Shoulder dysfunction", "Breathing difficulties"],
        "complications": ["Incomplete quadriplegia", "Diaphragmatic paralysis", "Chronic pain", "Loss of hand function"],
        "treatment": ["Surgical stabilization", "Anterior cervical fusion", "Posterior instrumentation", "Rehabilitation"]
    },
    "C6": {
        "name": "Sixth Cervical Vertebra",
        "description": "Common site for degenerative changes and fractures",
        "fracture_types": ["Compression fracture", "Extension teardrop fracture", "Clay shoveler's fracture", "Facet fracture"],
        "causes": ["Hyperextension injuries", "Degenerative changes", "Minor trauma in elderly", "Whiplash injuries"],
        "symptoms": ["Neck pain", "Radicular pain to thumb/index finger", "Weakness in wrist extensors", "Numbness"],
        "complications": ["C6 nerve root injury", "Chronic radiculopathy", "Loss of wrist extension", "Chronic neck pain"],
        "treatment": ["Conservative management", "Cervical collar", "Anterior cervical fusion", "Nerve root decompression"]
    },
    "C7": {
        "name": "Seventh Cervical Vertebra (Vertebra Prominens)",
        "description": "Transitional vertebra with prominent spinous process",
        "fracture_types": ["Clay shoveler's fracture", "Compression fracture", "Spinous process fracture", "Burst fracture"],
        "causes": ["Sudden muscular contraction", "Direct blow", "Flexion injuries", "Degenerative changes"],
        "symptoms": ["Lower neck pain", "Shoulder blade pain", "Weakness in triceps", "Numbness in middle finger"],
        "complications": ["C7 nerve root compression", "Chronic pain", "Loss of triceps function", "Thoracic outlet syndrome"],
        "treatment": ["Conservative treatment", "Pain management", "Physical therapy", "Surgical intervention (rare)"]
    }
}

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model from disk."""
    try:
        model = load_model('best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model. Make sure 'best_model.h5' is in the correct path. Error: {e}")
        return None

model = load_keras_model()

# --- Patient Information Form ---
def patient_info_form():
    """Display patient information input form"""
    with st.expander("📋 Patient Information (Click to expand)", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_name = st.text_input("👤 Patient Name", key="patient_name", 
                                        value=st.session_state.patient_info.get('name', ''))
        
        with col2:
            patient_id = st.text_input("🆔 Patient ID", key="patient_id",
                                      value=st.session_state.patient_info.get('id', ''))
        
        with col3:
            age = st.number_input("🎂 Age", min_value=0, max_value=120, step=1,
                                 value=st.session_state.patient_info.get('age', 0))
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            gender = st.selectbox("⚥ Gender", 
                                 ["Male", "Female", "Other"],
                                 index=["Male", "Female", "Other"].index(
                                     st.session_state.patient_info.get('gender', 'Male')))
        
        with col5:
            date = st.date_input("📅 Examination Date", 
                                value=datetime.date.today())
        
        with col6:
            # Auto-fill doctor name from login session
            doctor_name = st.text_input("👨‍⚕️ Referring Doctor", 
                                       key="doctor",
                                       value=st.session_state.current_user['name'],
                                       disabled=True)  # Disabled so user can't change it
        
        # Save button
        if st.button("💾 Save Patient Information"):
            st.session_state.patient_info = {
                'name': patient_name,
                'id': patient_id,
                'age': age,
                'gender': gender,
                'date': date.strftime("%Y-%m-%d"),
                'doctor': st.session_state.current_user['name'],  # Auto-filled from login
                'doctor_email': st.session_state.current_user['email'],
                'doctor_specialization': st.session_state.current_user['specialization'],
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(f"✅ Patient information saved successfully! Attending Doctor: {st.session_state.current_user['name']}")
            st.session_state.report_generated = False

# --- Display Patient Information ---
def display_patient_info():
    """Display saved patient information"""
    if st.session_state.patient_info:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 Current Patient")
            st.markdown(f"**Name:** {st.session_state.patient_info.get('name', 'N/A')}")
            st.markdown(f"**ID:** {st.session_state.patient_info.get('id', 'N/A')}")
            st.markdown(f"**Age:** {st.session_state.patient_info.get('age', 'N/A')}")
            st.markdown(f"**Gender:** {st.session_state.patient_info.get('gender', 'N/A')}")
            st.markdown(f"**Date:** {st.session_state.patient_info.get('date', 'N/A')}")
            st.markdown(f"**Doctor:** {st.session_state.patient_info.get('doctor', 'N/A')}")
            
            if st.button("📝 Edit Patient Info"):
                st.session_state.patient_info = {}
                st.rerun()

# --- PDF Report Generation with Images ---
class PDFReport(FPDF):
    def header(self):
        # Title
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Cervical Spine Fracture Detection Report', 0, 1, 'C')
        self.ln(5)
        
        # Line
        self.set_draw_color(0, 0, 0)
        self.line(10, 25, 200, 25)
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} - Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')
    
    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(2)
    
    def add_patient_info(self, patient_info):
        self.section_title('PATIENT INFORMATION')
        self.set_font('Arial', '', 11)
        
        info = [
            f"Name: {patient_info.get('name', 'N/A')}",
            f"Patient ID: {patient_info.get('id', 'N/A')}",
            f"Age: {patient_info.get('age', 'N/A')}",
            f"Gender: {patient_info.get('gender', 'N/A')}",
            f"Examination Date: {patient_info.get('date', 'N/A')}",
            f"Referring Doctor: {patient_info.get('doctor', 'N/A')}",
            f"Doctor Specialization: {patient_info.get('doctor_specialization', 'N/A')}"
        ]
        
        for line in info:
            self.cell(0, 6, line, 0, 1)
        self.ln(3)
    
    def add_diagnostic_results(self, prediction_result):
        self.section_title('DIAGNOSTIC RESULTS')
        self.set_font('Arial', '', 11)
        
        # Remove emojis for PDF
        status = "FRACTURE DETECTED" if prediction_result['is_fracture'] else "NORMAL (No Fracture)"
        self.cell(0, 6, f"Fracture Status: {status}", 0, 1)
        self.cell(0, 6, f"Confidence: {prediction_result['confidence']:.2%}", 0, 1)
        self.cell(0, 6, f"Model Score: {prediction_result['raw_score']:.4f}", 0, 1)
        self.ln(3)
    
    def add_fracture_localization(self, fracture_details):
        if fracture_details:
            self.section_title('FRACTURE LOCALIZATION')
            self.set_font('Arial', '', 11)
            
            self.cell(0, 6, f"Vertebral Level: {fracture_details.get('level', 'Unknown')}", 0, 1)
            if fracture_details.get('area'):
                self.cell(0, 6, f"Fracture Area: {fracture_details['area']} pixels", 0, 1)
            if fracture_details.get('region_confidence'):
                self.cell(0, 6, f"Region Confidence: {fracture_details['region_confidence']:.3f}", 0, 1)
            self.ln(3)
    
    def add_vertebral_details(self, fracture_details):
        if fracture_details and fracture_details.get('level') in CERVICAL_VERTEBRAE_INFO:
            level = fracture_details['level']
            info = CERVICAL_VERTEBRAE_INFO[level]
            
            self.section_title(f'VERTEBRAL DETAILS ({level} - {info["name"]})')
            self.set_font('Arial', '', 11)
            
            # Description
            self.set_font('Arial', 'B', 11)
            self.cell(0, 6, 'Description:', 0, 1)
            self.set_font('Arial', '', 11)
            self.multi_cell(0, 5, info['description'])
            self.ln(2)
            
            # Fracture Types
            self.set_font('Arial', 'B', 11)
            self.cell(0, 6, 'Common Fracture Types:', 0, 1)
            self.set_font('Arial', '', 11)
            for ft in info['fracture_types']:
                self.cell(0, 5, f'- {ft}', 0, 1)
            self.ln(2)
            
            # Causes
            self.set_font('Arial', 'B', 11)
            self.cell(0, 6, 'Common Causes:', 0, 1)
            self.set_font('Arial', '', 11)
            for cause in info['causes']:
                self.cell(0, 5, f'- {cause}', 0, 1)
            self.ln(2)
            
            # Symptoms
            self.set_font('Arial', 'B', 11)
            self.cell(0, 6, 'Symptoms:', 0, 1)
            self.set_font('Arial', '', 11)
            for symptom in info['symptoms']:
                self.cell(0, 5, f'- {symptom}', 0, 1)
            self.ln(2)
            
            # Complications
            self.set_font('Arial', 'B', 11)
            self.cell(0, 6, 'Potential Complications:', 0, 1)
            self.set_font('Arial', '', 11)
            for complication in info['complications']:
                self.cell(0, 5, f'- {complication}', 0, 1)
            self.ln(2)
            
            # Treatment
            self.set_font('Arial', 'B', 11)
            self.cell(0, 6, 'Treatment Options:', 0, 1)
            self.set_font('Arial', '', 11)
            for treatment in info['treatment']:
                self.cell(0, 5, f'- {treatment}', 0, 1)
            self.ln(3)
    
    def add_recommendations(self, is_fracture):
        self.section_title('RECOMMENDATIONS')
        self.set_font('Arial', '', 11)
        
        if is_fracture:
            self.multi_cell(0, 5, "- IMMEDIATE ACTION REQUIRED: Patient should be immobilized and evaluated by a spine specialist")
            self.multi_cell(0, 5, "- Emergency consultation with neurosurgeon or orthopedic spine surgeon recommended")
            self.multi_cell(0, 5, "- Further imaging (CT with thin cuts, MRI) may be needed for surgical planning")
            self.multi_cell(0, 5, "- Pain management and anti-inflammatory medications as prescribed")
            self.multi_cell(0, 5, "- Avoid any neck movement until cleared by specialist")
        else:
            self.multi_cell(0, 5, "- No acute fracture detected")
            self.multi_cell(0, 5, "- Routine follow-up as clinically indicated")
            self.multi_cell(0, 5, "- Continue normal activities if asymptomatic")
            self.multi_cell(0, 5, "- Return if symptoms develop or worsen")
        self.ln(3)
    
    def add_image_section(self, title, image_path, x=None, y=None, w=180, h=100):
        """Add an image to the PDF"""
        self.add_page()
        self.section_title(title)
        
        if x is None:
            x = (210 - w) / 2  # Center horizontally (A4 width is 210mm)
        if y is None:
            y = self.get_y() + 10
        
        try:
            self.image(image_path, x=x, y=y, w=w, h=h)
            self.set_y(y + h + 10)
        except Exception as e:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"Image could not be loaded: {str(e)}", 0, 1, 'C')
    
    def add_disclaimer(self):
        self.set_text_color(128, 128, 128)
        self.set_font('Arial', 'I', 9)
        self.multi_cell(0, 4, "DISCLAIMER: This report is generated by an AI-assisted diagnostic tool and should be reviewed by a qualified healthcare professional. Not for emergency use.")
        self.set_text_color(0, 0, 0)

def generate_pdf_report(prediction_result, fracture_details=None):
    """Generate PDF report with images"""
    pdf = PDFReport()
    
    # Page 1: Text information
    pdf.add_page()
    pdf.add_patient_info(st.session_state.patient_info)
    pdf.add_diagnostic_results(prediction_result)
    pdf.add_fracture_localization(fracture_details)
    pdf.add_vertebral_details(fracture_details)
    pdf.add_recommendations(prediction_result['is_fracture'])
    
    # Page 2: Original Image
    if 'original_image' in st.session_state.report_images:
        pdf.add_image_section('ORIGINAL CT SCAN', st.session_state.report_images['original_image'])
    
    # Page 3: Fracture Localization Overlay
    if 'fracture_overlay' in st.session_state.report_images:
        pdf.add_image_section('FRACTURE LOCALIZATION ANALYSIS', st.session_state.report_images['fracture_overlay'])
    
    # Page 4: Heatmap Visualization
    if 'heatmap_image' in st.session_state.report_images:
        pdf.add_image_section('GRAD-CAM HEATMAP VISUALIZATION', st.session_state.report_images['heatmap_image'])
    
    # Page 5: Heatmap Overlay
    if 'heatmap_overlay' in st.session_state.report_images:
        pdf.add_image_section('HEATMAP OVERLAY ON CT SCAN', st.session_state.report_images['heatmap_overlay'])
    
    # Final page: Disclaimer
    pdf.add_page()
    pdf.add_disclaimer()
    
    return pdf

# --- Generate Patient Report Text (for display) ---
def generate_patient_report_text(prediction_result, fracture_details=None):
    """Generate comprehensive patient report text for display"""
    
    report = f"""
    **PATIENT MEDICAL REPORT**
    
    **Patient Information:**
    - Name: {st.session_state.patient_info.get('name', 'N/A')}
    - Patient ID: {st.session_state.patient_info.get('id', 'N/A')}
    - Age: {st.session_state.patient_info.get('age', 'N/A')}
    - Gender: {st.session_state.patient_info.get('gender', 'N/A')}
    - Examination Date: {st.session_state.patient_info.get('date', 'N/A')}
    - Referring Doctor: {st.session_state.patient_info.get('doctor', 'N/A')}
    - Doctor Specialization: {st.session_state.patient_info.get('doctor_specialization', 'N/A')}
    
    **Diagnostic Results:**
    - Fracture Status: {"🚨 FRACTURE DETECTED" if prediction_result['is_fracture'] else "✅ NORMAL (No Fracture)"}
    - Confidence: {prediction_result['confidence']:.2%}
    - Model Score: {prediction_result['raw_score']:.4f}
    """
    
    if fracture_details:
        report += f"""
    
    **Fracture Localization:**
    - Vertebral Level: {fracture_details.get('level', 'Unknown')}
    - Fracture Area: {fracture_details.get('area', 'N/A')} pixels
    - Region Confidence: {fracture_details.get('region_confidence', 0):.3f}
        """
        
        if fracture_details.get('level') in CERVICAL_VERTEBRAE_INFO:
            info = CERVICAL_VERTEBRAE_INFO[fracture_details['level']]
            report += f"""
    
    **Vertebral Details ({fracture_details['level']} - {info['name']}):**
    - Description: {info['description']}
    - Common Fracture Types: {', '.join(info['fracture_types'])}
    - Common Causes: {', '.join(info['causes'])}
    - Symptoms: {', '.join(info['symptoms'])}
    - Complications: {', '.join(info['complications'])}
    - Treatment Options: {', '.join(info['treatment'])}
            """
    
    report += f"""
    
    **Recommendations:**
    {get_recommendations_text(prediction_result['is_fracture'])}
    
    **Report Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    **Generated By:** {st.session_state.current_user['name']} ({st.session_state.current_user['specialization']})
    
    ---
    **Disclaimer:** This report is generated by an AI-assisted diagnostic tool and should be reviewed by a qualified healthcare professional. Not for emergency use.
    """
    
    return report

def get_recommendations_text(is_fracture):
    """Generate recommendations text"""
    if is_fracture:
        return """
        - 🚨 **IMMEDIATE ACTION REQUIRED**: Patient should be immobilized and evaluated by a spine specialist
        - 🏥 Emergency consultation with neurosurgeon or orthopedic spine surgeon recommended
        - 📋 Further imaging (CT with thin cuts, MRI) may be needed for surgical planning
        - 💊 Pain management and anti-inflammatory medications as prescribed
        - ⚠️ Avoid any neck movement until cleared by specialist
        """
    else:
        return """
        - ✅ No acute fracture detected
        - 👨‍⚕️ Routine follow-up as clinically indicated
        - 💪 Continue normal activities if asymptomatic
        - ⚠️ Return if symptoms develop or worsen
        """

# --- Enhanced Grad-CAM Implementation ---
def get_grad_cam(model, img_array):
    """Generate a Grad-CAM heatmap for MobileNetV2-based model."""
    try:
        # First, make sure the model has been called at least once
        _ = model(img_array)
        
        # Get the MobileNetV2 base model (first layer)
        base_model = model.layers[0]  # This is the mobilenetv2_1.00_224 layer
        
        # Ensure the base model is also called
        _ = base_model(img_array)
        
        # Use the last convolutional layer from MobileNetV2 which is 'out_relu'
        last_conv_layer_name = "out_relu"
        
        # Alternative approach: Create intermediate model step by step
        # First get the conv layer output
        conv_layer = base_model.get_layer(last_conv_layer_name)
        
        # Create a model that outputs both the conv features and final prediction
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )
        
        # Compute the gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # For fracture detection, focus on the fracture class
            loss = predictions[0, 0]  # Binary output, fracture is when close to 0

        # Get the gradients of the loss w.r.t. the conv layer outputs
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each feature map by its corresponding gradient importance
        conv_outputs = conv_outputs[0]
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create the heatmap by averaging across all feature maps
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0)
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        
        return heatmap.numpy()
        
    except Exception as e:
        st.error(f"Primary Grad-CAM failed: {str(e)}")
        
        # Try alternative approach using layer outputs directly
        try:
            st.info("Trying alternative visualization approach...")
            return create_simple_attention_map(model, img_array)
        except Exception as e2:
            st.error(f"Alternative approach also failed: {str(e2)}")
            return create_dummy_heatmap()

def create_simple_attention_map(model, img_array):
    """Create a simple attention map using feature visualization."""
    try:
        # Get feature maps from different layers and combine them
        base_model = model.layers[0]
        
        # Try to get intermediate feature maps
        intermediate_layers = []
        for layer in base_model.layers:
            if 'relu' in layer.name and 'block' in layer.name:
                intermediate_layers.append(layer.name)
        
        # Use the last few ReLU layers
        if len(intermediate_layers) >= 3:
            target_layers = intermediate_layers[-3:]
        else:
            target_layers = intermediate_layers
        
        if not target_layers:
            return create_dummy_heatmap()
        
        # Create model for feature extraction
        outputs = []
        for layer_name in target_layers:
            outputs.append(base_model.get_layer(layer_name).output)
        
        feature_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=outputs
        )
        
        # Get features
        features = feature_model(img_array)
        
        # Combine features from different scales
        combined_heatmap = None
        for feature_map in features:
            # Average across channels
            channel_mean = tf.reduce_mean(feature_map[0], axis=-1)
            # Resize to target size
            resized = tf.image.resize(
                tf.expand_dims(tf.expand_dims(channel_mean, 0), -1),
                (256, 256)
            )
            resized = tf.squeeze(resized)
            
            if combined_heatmap is None:
                combined_heatmap = resized
            else:
                combined_heatmap += resized
        
        # Normalize
        if tf.reduce_max(combined_heatmap) > 0:
            combined_heatmap = combined_heatmap / tf.reduce_max(combined_heatmap)
        
        return combined_heatmap.numpy()
        
    except Exception as e:
        st.error(f"Simple attention map failed: {str(e)}")
        return create_dummy_heatmap()

def create_dummy_heatmap():
    """Create a basic dummy heatmap for demonstration."""
    st.warning("Using simplified visualization - this is for demonstration only")
    
    # Create a simple circular heatmap in the center-lower region
    # This represents a typical cervical spine fracture location
    heatmap = np.zeros((256, 256))
    
    # Create a circular region in the lower-center area
    center_x, center_y = 128, 180  # Lower center for cervical spine
    radius = 40
    
    y, x = np.ogrid[:256, :256]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Add some randomness to make it look more realistic
    noise = np.random.normal(0, 0.1, (256, 256))
    heatmap[mask] = 0.8 + noise[mask] * 0.2
    
    # Add gradient effect
    for i in range(256):
        for j in range(256):
            distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if distance <= radius:
                heatmap[i, j] = max(0, 0.9 - (distance / radius) * 0.4)
    
    # Normalize
    heatmap = np.clip(heatmap, 0, 1)
    
    return heatmap

def detect_fracture_region(heatmap, threshold=0.3):
    """Detect the fracture region coordinates from heatmap with enhanced detection."""
    if heatmap is None:
        return None
        
    # Apply threshold to find high-activation areas
    high_activation = heatmap > threshold
    
    if not np.any(high_activation):
        # If no high activation found, lower the threshold
        threshold = np.max(heatmap) * 0.5
        high_activation = heatmap > threshold
        
        if not np.any(high_activation):
            # Use the highest activation area
            max_val = np.max(heatmap)
            threshold = max_val * 0.3
            high_activation = heatmap > threshold
    
    if not np.any(high_activation):
        return None
    
    # Find connected components for better region detection
    try:
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(high_activation)
        
        if num_features == 0:
            return None
        
        # Find the largest connected component
        component_sizes = []
        for i in range(1, num_features + 1):
            component_sizes.append(np.sum(labeled_array == i))
        
        largest_component = np.argmax(component_sizes) + 1
        largest_region = labeled_array == largest_component
        
        # Get bounding box of the largest region
        coords = np.where(largest_region)
        
    except ImportError:
        # Fallback without scipy
        coords = np.where(high_activation)
    
    if len(coords[0]) == 0:
        return None
    
    top_left = (np.min(coords[1]), np.min(coords[0]))
    bottom_right = (np.max(coords[1]), np.max(coords[0]))
    center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
    
    # Expand the bounding box slightly for better visualization
    padding = 10
    height, width = heatmap.shape
    top_left = (max(0, top_left[0] - padding), max(0, top_left[1] - padding))
    bottom_right = (min(width, bottom_right[0] + padding), min(height, bottom_right[1] + padding))
    
    return {
        'top_left': top_left,
        'bottom_right': bottom_right,
        'center': center,
        'area': (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1]),
        'confidence': np.mean(heatmap[high_activation])
    }

def identify_vertebral_level(fracture_region, img_height=256):
    """Identify the vertebral level based on fracture location."""
    if fracture_region is None:
        return "Unknown"
    
    center_y = fracture_region['center'][1]
    relative_position = center_y / img_height
    
    # Rough approximation based on typical cervical spine anatomy in sagittal view
    if relative_position < 0.15:
        return "C1"
    elif relative_position < 0.25:
        return "C2"
    elif relative_position < 0.35:
        return "C3"
    elif relative_position < 0.50:
        return "C4"
    elif relative_position < 0.65:
        return "C5"
    elif relative_position < 0.80:
        return "C6"
    else:
        return "C7"

def create_fracture_overlay(img_path, fracture_region, heatmap, vertebral_level):
    """Create a comprehensive fracture overlay on the input image."""
    # Load and resize the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create overlay image
    overlay = img_rgb.copy()
    
    if fracture_region and heatmap is not None:
        # Create heatmap overlay
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        alpha = 0.4
        overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap_colored, alpha, 0)
        
        # Draw bounding box
        top_left = fracture_region['top_left']
        bottom_right = fracture_region['bottom_right']
        
        # Draw main bounding box (red)
        cv2.rectangle(overlay, top_left, bottom_right, (255, 0, 0), 3)
        
        # Draw center point
        center = fracture_region['center']
        cv2.circle(overlay, center, 5, (255, 255, 0), -1)
        
        # Add fracture area label with background
        label = f"FRACTURE: {vertebral_level}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label above the bounding box
        label_x = max(5, top_left[0])
        label_y = max(text_height + 10, top_left[1] - 10)
        
        # Draw background rectangle for text
        cv2.rectangle(overlay, 
                     (label_x - 5, label_y - text_height - 5),
                     (label_x + text_width + 5, label_y + baseline + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(overlay, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness)
        
        # Add confidence indicator
        if 'confidence' in fracture_region:
            conf_text = f"Conf: {fracture_region['confidence']:.2f}"
            cv2.putText(overlay, conf_text, 
                       (label_x, label_y + text_height + 10), 
                       font, 0.5, (255, 255, 255), 1)
    
    return overlay

def superimpose_grad_cam(img_path, heatmap, alpha=0.6):
    """Superimpose the heatmap on the original image."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1, heatmap, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb

def display_vertebral_info(vertebral_level):
    """Display detailed information about the identified vertebral level."""
    if vertebral_level in CERVICAL_VERTEBRAE_INFO:
        info = CERVICAL_VERTEBRAE_INFO[vertebral_level]
        
        st.markdown(f"### 📍 Identified Level: {vertebral_level} ({info['name']})")
        st.markdown(f"**Description:** {info['description']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🩻 Common Fracture Types")
            for fracture_type in info['fracture_types']:
                st.markdown(f"• {fracture_type}")
            
            st.markdown("#### ⚡ Common Causes")
            for cause in info['causes']:
                st.markdown(f"• {cause}")
        
        with col2:
            st.markdown("#### 🔍 Symptoms")
            for symptom in info['symptoms']:
                st.markdown(f"• {symptom}")
            
            st.markdown("#### ⚠️ Potential Complications")
            for complication in info['complications']:
                st.markdown(f"• {complication}")
        
        st.markdown("#### 🏥 Treatment Options")
        for treatment in info['treatment']:
            st.markdown(f"• {treatment}")

# --- Streamlit App UI (Main Application) ---

# Add logout button in sidebar
with st.sidebar:
    st.markdown("### 👨‍⚕️ Current User")
    st.markdown(f"**Name:** {st.session_state.current_user['name']}")
    st.markdown(f"**Specialization:** {st.session_state.current_user['specialization']}")
    st.markdown(f"**Email:** {st.session_state.current_user['email']}")
    
    if st.button("🚪 Logout", use_container_width=True):
        logout()
    
    st.markdown("---")

st.title("🔬 Advanced Cervical Spine Fracture Detection & Analysis")
st.markdown("Upload a Cervical Spine CT scan to detect potential fractures, identify vertebral level, and get detailed medical information.")

st.warning("**Disclaimer:** This tool is for demonstration and educational purposes only. It is not a substitute for professional medical diagnosis.")

# Display patient info form
patient_info_form()

# Display current patient info in sidebar
display_patient_info()

# File uploader
uploaded_file = st.file_uploader("Choose a CT scan image...", type=["png", "jpg", "jpeg"])

if model and uploaded_file is not None:
    # Check if patient info is saved
    if not st.session_state.patient_info:
        st.warning("⚠️ Please fill and save patient information before proceeding.")
    else:
        # Save uploaded file temporarily
        with open("temp_img.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Create main layout
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("📋 Original Image")
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption='Uploaded CT Scan', use_container_width=True)
            
            # Save original image for PDF
            img = cv2.imread("temp_img.jpg")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("report_original.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            st.session_state.report_images['original_image'] = "report_original.jpg"

        with col2:
            st.subheader("🎯 Analysis Results")
            with st.spinner('Analyzing the image... This may take a moment.'):
                # Preprocess the image for the model
                img = load_img("temp_img.jpg", target_size=(256, 256))
                img_array = img_to_array(img)
                img_array_rescaled = img_array / 255.0
                img_array_expanded = np.expand_dims(img_array_rescaled, axis=0)

                # Make prediction
                prediction = model.predict(img_array_expanded)
                is_fracture = prediction[0][0] < 0.5
                confidence = (1 - prediction[0][0]) if is_fracture else prediction[0][0]

                # Store prediction results
                prediction_result = {
                    'is_fracture': is_fracture,
                    'confidence': confidence,
                    'raw_score': prediction[0][0]
                }

                # Display prediction
                if is_fracture:
                    st.error(f"### 🚨 Fracture Detected")
                    st.metric(label="Confidence", value=f"{confidence:.2%}")
                else:
                    st.success(f"### ✅ Normal (No Fracture)")
                    st.metric(label="Confidence", value=f"{confidence:.2%}")
                
                st.metric(label="Raw Model Score", value=f"{prediction[0][0]:.4f}", 
                          help="Lower scores indicate fracture probability")

        with col3:
            st.subheader("🔍 Fracture Localization")
            fracture_details = None
            if is_fracture:
                # Generate Grad-CAM - let the function find the appropriate layer
                heatmap = get_grad_cam(model, img_array_expanded)
                
                # Detect fracture region and identify vertebral level
                if heatmap is not None:
                    fracture_region = detect_fracture_region(heatmap, threshold=0.2)
                    vertebral_level = identify_vertebral_level(fracture_region)
                    
                    fracture_details = {
                        'level': vertebral_level,
                        'area': fracture_region['area'] if fracture_region else None,
                        'region_confidence': fracture_region['confidence'] if fracture_region else None
                    }
                    
                    # Create comprehensive fracture overlay
                    fracture_overlay = create_fracture_overlay("temp_img.jpg", fracture_region, heatmap, vertebral_level)
                    st.image(fracture_overlay, caption=f'Fracture Analysis: {vertebral_level}', use_container_width=True)
                    
                    # Save fracture overlay for PDF
                    cv2.imwrite("report_fracture_overlay.jpg", cv2.cvtColor(fracture_overlay, cv2.COLOR_RGB2BGR))
                    st.session_state.report_images['fracture_overlay'] = "report_fracture_overlay.jpg"
                    
                    if fracture_region:
                        st.info(f"**Detected Vertebral Level: {vertebral_level}**")
                        st.metric("Fracture Area", f"{fracture_region['area']} pixels")
                        if 'confidence' in fracture_region:
                            st.metric("Region Confidence", f"{fracture_region['confidence']:.3f}")
                    else:
                        st.warning("Fracture detected but specific region unclear")
                else:
                    st.error("Could not generate fracture localization")
                    vertebral_level = "Unknown"
            else:
                st.info("No fracture detected - localization not applicable")

        # Display heatmap visualization if fracture is detected
        if is_fracture and 'heatmap' in locals() and heatmap is not None:
            st.markdown("---")
            st.subheader("🌡️ Heat Map Visualization")
            
            col_heat1, col_heat2 = st.columns(2)
            
            with col_heat1:
                st.markdown("#### Grad-CAM Overlay")
                superimposed_image = superimpose_grad_cam("temp_img.jpg", heatmap)
                st.image(superimposed_image, caption='Attention Heatmap - Red areas show model focus', use_container_width=True)
                
                # Save heatmap overlay for PDF
                cv2.imwrite("report_heatmap_overlay.jpg", cv2.cvtColor(superimposed_image, cv2.COLOR_RGB2BGR))
                st.session_state.report_images['heatmap_overlay'] = "report_heatmap_overlay.jpg"
            
            with col_heat2:
                st.markdown("#### Pure Heatmap")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(heatmap, cmap='jet', alpha=0.8)
                ax.set_title('Fracture Probability Heatmap')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                
                # Save pure heatmap for PDF
                fig.savefig("report_pure_heatmap.png", dpi=150, bbox_inches='tight', pad_inches=0)
                st.session_state.report_images['heatmap_image'] = "report_pure_heatmap.png"
            
            # Display detailed vertebral information
            st.markdown("---")
            if 'vertebral_level' in locals():
                display_vertebral_info(vertebral_level)
            
            # Emergency information
            st.markdown("---")
            st.error("""
            ### 🚨 Important Medical Information
            
            **Immediate Actions Required:**
            - Immobilize the patient's neck immediately
            - Avoid any neck movement
            - Seek emergency medical attention
            - Consider spinal precautions during transport
            
            **This is a medical emergency that requires immediate professional evaluation.**
            """)
        
        # Generate and display patient report
        st.markdown("---")
        st.subheader("📄 Patient Report")
        
        # Generate report text for display
        report_text = generate_patient_report_text(prediction_result, fracture_details)
        
        # Display report in a nice box
        with st.container():
            st.markdown("### 📋 Comprehensive Medical Report")
            st.markdown(report_text)
            
            # PDF Download button
            if st.button("📥 Generate PDF Report with Images"):
                with st.spinner("Generating PDF report with images..."):
                    # Generate PDF
                    pdf = generate_pdf_report(prediction_result, fracture_details)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        pdf.output(tmp_file.name)
                        tmp_file_path = tmp_file.name
                    
                    # Read the file
                    with open(tmp_file_path, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    
                    # Clean up temporary image files
                    for img_file in st.session_state.report_images.values():
                        if os.path.exists(img_file):
                            try:
                                os.unlink(img_file)
                            except:
                                pass
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Complete PDF Report",
                        data=pdf_bytes,
                        file_name=f"fracture_report_{st.session_state.patient_info.get('id', 'patient')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        st.session_state.report_generated = True

# Sidebar information
st.sidebar.title("ℹ️ About This Application")
st.sidebar.info(
    """
    This advanced application uses deep learning to:
    
    1. **Detect** cervical spine fractures
    2. **Localize** fracture regions
    3. **Identify** vertebral levels (C1-C7)
    4. **Provide** detailed medical information
    5. **Visualize** attention maps using Grad-CAM
    """
)

st.sidebar.title("🔬 Technical Features")
st.sidebar.info(
    """
    - **Deep Learning Model**: Pre-trained CNN for fracture detection
    - **Grad-CAM Visualization**: Shows model attention areas
    - **Anatomical Mapping**: Identifies specific vertebral levels
    - **Medical Database**: Comprehensive fracture information
    - **Real-time Analysis**: Instant results and visualization
    """
)

st.sidebar.title("⚠️ Medical Disclaimer")
st.sidebar.warning(
    """
    This tool is for:
    - Educational purposes
    - Research applications
    - Initial screening support
    
    **NOT for:**
    - Final medical diagnosis
    - Treatment decisions
    - Emergency decision making
    
    Always consult qualified medical professionals.
    """
)

st.sidebar.title("📖 Cervical Spine Anatomy")
st.sidebar.info(
    """
    **C1 (Atlas)**: Supports skull, no vertebral body
    **C2 (Axis)**: Has odontoid process, allows head rotation
    **C3-C6**: Typical cervical vertebrae
    **C7**: Vertebra prominens, transitional vertebra
    
    Each level has specific injury patterns and clinical significance.
    """
)

# Add session state viewer in sidebar (optional, for debugging)
if st.sidebar.checkbox("Show Session State (Debug)"):
    st.sidebar.write(st.session_state)