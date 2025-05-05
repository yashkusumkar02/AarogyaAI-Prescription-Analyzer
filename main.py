import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import google.generativeai as genai
from gtts import gTTS
import os
from io import BytesIO
import re
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
AudioSegment.converter = "path_to_your_ffmpeg"  # Set this if you have ffmpeg in a non-standard location

# Load environment variables
load_dotenv()

# --------------------------
# Configuration
# --------------------------
st.set_page_config(
    page_title="AarogyaAI Pro - Advanced Prescription Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services with error handling
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        st.error("Missing Gemini API key in .env file")
        st.stop()
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Using Gemini 1.5 Flash model
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    medical_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    chat_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    # Set Tesseract path
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
except Exception as e:
    st.error(f"Service initialization failed: {str(e)}")
    st.stop()

# --------------------------
# Core Functions
# --------------------------
def text_to_speech(text, lang='en'):
    """Convert text to speech with error handling"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"Voice generation failed: {str(e)}")
        return None

def speech_to_text(lang='en'):
    """Convert speech to text with error handling"""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now")
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio, language=lang)
            return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.warning(f"Could not request results; {e}")
        return None
    except Exception as e:
        st.warning(f"Voice recognition failed: {str(e)}")
        return None

def load_and_validate_image(uploaded_file):
    """Load and validate the uploaded image file"""
    try:
        img = Image.open(uploaded_file)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        return img
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None

def enhance_image(image):
    """Advanced image preprocessing for OCR"""
    try:
        img = np.array(image)
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1,1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(processed)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return image

def translate_text(text, target_lang='en'):
    """Translate text with error handling"""
    try:
        if target_lang == 'en':
            return text
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}")
        return text

# Complete UI translations for Indian languages
UI_TRANSLATIONS = {
    "English": {
        "title": "🏥 AarogyaAI Advanced Prescription Analyzer",
        "patient_profile": "Patient Profile",
        "input_method": "Input Method",
        "image_upload": "Image Upload",
        "text_input": "Text Input",
        "age": "Age",
        "weight": "Weight (kg)",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "other": "Other",
        "allergies": "Allergies",
        "conditions": "Existing Conditions",
        "output_language": "Application Language",
        "prescription_input": "Prescription Input",
        "upload_prescription": "Upload Prescription Image",
        "upload_help": "For best results, use clear, well-lit images",
        "extracted_text": "Extracted Text",
        "enter_prescription": "Enter Prescription Text",
        "enter_placeholder": "Enter prescription details here...",
        "analyze_btn": "Analyze Prescription",
        "no_prescription": "Please enter or upload a prescription",
        "extracting": "Extracting text from image...",
        "no_text": "Could not extract text. Please try a clearer image or use text input.",
        "analyzing": "Analyzing with medical AI...",
        "analysis_results": "Analysis Results",
        "chat_assistant": "Prescription Chat Assistant",
        "chat_prompt": "Ask questions about your analyzed prescription",
        "chat_input": "Ask about your prescription...",
        "voice_input": "🎤 Voice Input",
        "profile_info": "Patient profile will be enabled for text input",
        "voice_instructions": "Click the microphone and speak your question",
        "general_health": "General Health Information",
        "medication_info": "Medication Information",
        "side_effects": "Side Effects",
        "diet_advice": "Diet Advice",
        "exercise_advice": "Exercise Advice",
        "emergency_info": "Emergency Information",
        "clear_chat": "Clear Chat History",
        "prescription_context": "Prescription Context",
        "patient_context": "Patient Context",
        "system_prompt": "System Instructions",
        "chat_history": "Conversation History"
    },
    "Hindi": {
        "title": "🏥 आरोग्यएआई उन्नत प्रिस्क्रिप्शन विश्लेषक",
        "patient_profile": "रोगी प्रोफ़ाइल",
        "input_method": "इनपुट विधि",
        "image_upload": "छवि अपलोड",
        "text_input": "पाठ इनपुट",
        "age": "आयु",
        "weight": "वजन (किलो)",
        "gender": "लिंग",
        "male": "पुरुष",
        "female": "महिला",
        "other": "अन्य",
        "allergies": "एलर्जी",
        "conditions": "मौजूदा स्थितियाँ",
        "output_language": "एप्लिकेशन भाषा",
        "prescription_input": "प्रिस्क्रिप्शन इनपुट",
        "upload_prescription": "प्रिस्क्रिप्शन छवि अपलोड करें",
        "upload_help": "सर्वोत्तम परिणामों के लिए स्पष्ट, अच्छी रोशनी वाली छवियों का उपयोग करें",
        "extracted_text": "निकाला गया पाठ",
        "enter_prescription": "प्रिस्क्रिप्शन पाठ दर्ज करें",
        "enter_placeholder": "प्रिस्क्रिप्शन विवरण यहाँ दर्ज करें...",
        "analyze_btn": "प्रिस्क्रिप्शन का विश्लेषण करें",
        "no_prescription": "कृपया प्रिस्क्रिप्शन दर्ज या अपलोड करें",
        "extracting": "छवि से पाठ निकाला जा रहा है...",
        "no_text": "पाठ निकाला नहीं जा सका। कृपया स्पष्ट छवि का प्रयास करें या पाठ इनपुट का उपयोग करें।",
        "analyzing": "चिकित्सा एआई के साथ विश्लेषण किया जा रहा है...",
        "analysis_results": "विश्लेषण परिणाम",
        "chat_assistant": "प्रिस्क्रिप्शन चैट सहायक",
        "chat_prompt": "विश्लेषित प्रिस्क्रिप्शन के बारे में प्रश्न पूछें",
        "chat_input": "अपने प्रिस्क्रिप्शन के बारे में पूछें...",
        "voice_input": "🎤 वॉइस इनपुट",
        "profile_info": "पाठ इनपुट के लिए रोगी प्रोफ़ाइल सक्षम होगा",
        "voice_instructions": "माइक्रोफोन पर क्लिक करें और अपना प्रश्न बोलें",
        "general_health": "सामान्य स्वास्थ्य जानकारी",
        "medication_info": "दवा की जानकारी",
        "side_effects": "दुष्प्रभाव",
        "diet_advice": "आहार सलाह",
        "exercise_advice": "व्यायाम सलाह",
        "emergency_info": "आपातकालीन जानकारी",
        "clear_chat": "चैट इतिहास साफ करें",
        "prescription_context": "प्रिस्क्रिप्शन संदर्भ",
        "patient_context": "रोगी संदर्भ",
        "system_prompt": "सिस्टम निर्देश",
        "chat_history": "वार्तालाप इतिहास"
    }
}

# Language mapping for gTTS and translation
LANGUAGE_MAPPING = {
    "English": {"code": "en", "display": "English"},
    "Hindi": {"code": "hi", "display": "हिंदी"},
    "Marathi": {"code": "mr", "display": "मराठी"},
    "Tamil": {"code": "ta", "display": "தமிழ்"},
    "Telugu": {"code": "te", "display": "తెలుగు"},
    "Kannada": {"code": "kn", "display": "ಕನ್ನಡ"},
    "Malayalam": {"code": "ml", "display": "മലയാളം"},
    "Bengali": {"code": "bn", "display": "বাংলা"},
    "Gujarati": {"code": "gu", "display": "ગુજરાતી"},
    "Punjabi": {"code": "pa", "display": "ਪੰਜਾਬੀ"},
    "Odia": {"code": "or", "display": "ଓଡ଼ିଆ"},
    "Urdu": {"code": "ur", "display": "اردو"},
    "Assamese": {"code": "as", "display": "অসমীয়া"},
    "Nepali": {"code": "ne", "display": "नेपाली"},
    "Sanskrit": {"code": "sa", "display": "संस्कृतम्"},
    "French": {"code": "fr", "display": "Français"},
    "Spanish": {"code": "es", "display": "Español"},
    "German": {"code": "de", "display": "Deutsch"},
    "Chinese": {"code": "zh", "display": "中文"},
    "Japanese": {"code": "ja", "display": "日本語"},
    "Russian": {"code": "ru", "display": "Русский"},
    "Arabic": {"code": "ar", "display": "العربية"}
}

HEALTH_QUESTIONS = {
    "English": [
        "What are the side effects of this medication?",
        "Can I take this with alcohol?",
        "What food should I avoid with these medicines?",
        "When should I take this medicine?",
        "What are the warning signs to watch for?"
    ],
    "Hindi": [
        "इस दवा के साइड इफेक्ट क्या हैं?",
        "क्या मैं इसे शराब के साथ ले सकता हूँ?",
        "इन दवाओं के साथ मुझे कौन से खाद्य पदार्थों से बचना चाहिए?",
        "मुझे यह दवा कब लेनी चाहिए?",
        "मुझे किन चेतावनी संकेतों पर नजर रखनी चाहिए?"
    ]
}

# --------------------------
# Main Application
# --------------------------
def main():
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'voice_input' not in st.session_state:
        st.session_state.voice_input = False
    if 'prescription_text' not in st.session_state:
        st.session_state.prescription_text = ""
    if 'patient_context' not in st.session_state:
        st.session_state.patient_context = {}
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = ""

    # Language selection at the very top
    lang_col, _ = st.columns([1, 4])
    with lang_col:
        selected_lang = st.selectbox(
            "🌐 Select Application Language / एप्लिकेशन भाषा चुनें",
            sorted(UI_TRANSLATIONS.keys()),
            key='app_language'
        )
    
    # Get translations for selected language
    t = UI_TRANSLATIONS.get(selected_lang, UI_TRANSLATIONS["English"])
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
        .stApp {
            background-color: #f5f9ff;
        }
        .stSidebar {
            background-color: #e6f0ff !important;
        }
        .stButton button {
            background-color: #4a90e2;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stButton button:hover {
            background-color: #3a7bc8;
            color: white;
        }
        .stTextInput input, .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #4a90e2;
        }
        .stSelectbox select {
            border-radius: 8px;
            border: 1px solid #4a90e2;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e6f0ff;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #ffffff;
            margin-right: 20%;
        }
        .analysis-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .prescription-card {
            background-color: #f0f7ff;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4a90e2;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title(t["title"])
    
    # Sidebar - Patient Profile (only shown for text input)
    with st.sidebar:
        st.header(t["patient_profile"])
        
        input_method = st.radio(
            t["input_method"],
            [t["image_upload"], t["text_input"]],
            index=0,
            key='input_method'
        )
        
        if input_method == t["text_input"]:
            age = st.number_input(t["age"], 1, 120, 30)
            weight = st.number_input(t["weight"], 30, 200, 70)
            gender = st.radio(t["gender"], [t["male"], t["female"], t["other"]])
            allergies = st.multiselect(
                t["allergies"], 
                ["Penicillin", "Sulfa", "NSAIDs", "Iodine", "Latex", "None"]
            )
            conditions = st.multiselect(
                t["conditions"],
                ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "None"]
            )
        else:
            st.info(t["profile_info"])
            age, weight, gender, allergies, conditions = 30, 70, t["male"], [], []
        
        # Language selection for output (always visible)
        output_lang = st.selectbox(
            t["output_language"],
            sorted(LANGUAGE_MAPPING.keys()),
            key='output_language'
        )
        
        # Common health questions
        st.subheader(t["general_health"])
        if st.button(t["medication_info"]):
            st.session_state.chat_history.append({
                "role": "user",
                "content": HEALTH_QUESTIONS.get(output_lang, HEALTH_QUESTIONS["English"])[0]
            })
            st.rerun()
        if st.button(t["side_effects"]):
            st.session_state.chat_history.append({
                "role": "user",
                "content": HEALTH_QUESTIONS.get(output_lang, HEALTH_QUESTIONS["English"])[1]
            })
            st.rerun()
        if st.button(t["diet_advice"]):
            st.session_state.chat_history.append({
                "role": "user",
                "content": HEALTH_QUESTIONS.get(output_lang, HEALTH_QUESTIONS["English"])[2]
            })
            st.rerun()
        
        # Clear chat button
        if st.button(t["clear_chat"]):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader(t["prescription_input"])
        prescription_text = ""
        
        if input_method == t["image_upload"]:
            uploaded_file = st.file_uploader(
                t["upload_prescription"],
                type=["jpg", "jpeg", "png", "pdf"],
                help=t["upload_help"]
            )
            
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    st.warning("PDF processing coming soon. Please upload an image file.")
                else:
                    img = load_and_validate_image(uploaded_file)
                    if img:
                        st.image(img, use_container_width=True)
                        with st.spinner(t["extracting"]):
                            enhanced = enhance_image(img)
                            prescription_text = pytesseract.image_to_string(enhanced)
                            st.session_state.prescription_text = prescription_text
                            if prescription_text.strip():
                                st.markdown(f"""
                                <div class="prescription-card">
                                    <strong>{t['extracted_text']}:</strong><br>
                                    {prescription_text}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning(t["no_text"])
        
        else:  # Text Input
            prescription_text = st.text_area(
                t["enter_prescription"],
                height=200,
                placeholder=t["enter_placeholder"],
                key='prescription_text_input'
            )
            st.session_state.prescription_text = prescription_text
        
        # Analysis button
        if st.button(t["analyze_btn"], type="primary", key='analyze_btn'):
            if not st.session_state.prescription_text.strip():
                st.warning(t["no_prescription"])
            else:
                with st.spinner(t["analyzing"]):
                    try:
                        # Prepare context
                        lang_code = LANGUAGE_MAPPING[output_lang]["code"]
                        context = {
                            'age': age,
                            'weight': weight,
                            'gender': gender,
                            'allergies': allergies,
                            'conditions': conditions,
                            'language': lang_code
                        }
                        st.session_state.patient_context = context
                        
                        # Generate prompt
                        prompt = f"""
                        Analyze this prescription for a {context['age']} year old {context['gender']}:
                        
                        **Patient Details:**
                        - Weight: {context['weight']} kg
                        - Allergies: {', '.join(context['allergies']) or 'None'}
                        - Conditions: {', '.join(context['conditions']) or 'None'}
                        
                        **Prescription:**
                        {st.session_state.prescription_text}
                        
                        Provide comprehensive analysis in {output_lang} covering:
                        1. All medications identified with dosage information
                        2. Verification if dosage is age/weight appropriate
                        3. Potential drug interactions
                        4. Possible allergy/condition conflicts
                        5. Clear patient instructions in simple terms
                        6. Red flags requiring immediate medical attention
                        
                        Format your response with:
                        - Clear section headings
                        - Bullet points for readability
                        - Relevant emojis for visual cues
                        - Warning symbols for critical information
                        """
                        
                        # Get analysis using Gemini 1.5 Flash
                        response = medical_model.generate_content(prompt)
                        analysis = response.text
                        
                        # Translate if needed
                        if lang_code != 'en':
                            analysis = translate_text(analysis, lang_code)
                        
                        # Store analysis in session state
                        st.session_state.analysis_result = analysis
                        st.session_state.analysis_done = True
                        
                        # Rerun to update the display
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        
        # Display analysis results if available
        if st.session_state.get('analysis_done', False):
            st.subheader(f"{t['analysis_results']} ({output_lang})")
            
            # Create a nice card for the analysis
            with st.container():
                st.markdown(f"""
                <div class="analysis-card">
                    {st.session_state.analysis_result}
                </div>
                """, unsafe_allow_html=True)
                
                # Voice output
                if len(st.session_state.analysis_result) > 0:
                    lang_code = LANGUAGE_MAPPING[output_lang]["code"]
                    audio = text_to_speech(
                        st.session_state.analysis_result[:500],  # First 500 chars
                        lang_code
                    )
                    if audio:
                        st.audio(audio, format='audio/mp3')
    
    # Enhanced Chat interface (right column)
    with col2:
        st.subheader(t["chat_assistant"])
        
        # Display context information
        if st.session_state.get('analysis_done', False):
            with st.expander(t["prescription_context"]):
                st.write(f"**{t['patient_context']}**")
                st.json(st.session_state.patient_context)
                st.write("**Prescription Excerpt:**")
                st.text(st.session_state.prescription_text[:200] + "...")
        
        st.info(t["chat_prompt"])
        
        # Voice input button
        if st.button(t["voice_input"]):
            st.session_state.voice_input = True
        
        if st.session_state.get('voice_input', False):
            lang_code = LANGUAGE_MAPPING[output_lang]["code"]
            voice_text = speech_to_text(lang_code)
            if voice_text:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": voice_text
                })
                st.session_state.voice_input = False
                st.rerun()
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AarogyaAI:</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Add voice output for assistant messages
                if msg["role"] == "assistant":
                    lang_code = LANGUAGE_MAPPING[output_lang]["code"]
                    audio = text_to_speech(msg["content"], lang_code)
                    if audio:
                        st.audio(audio, format='audio/mp3')
        
        # Chat input
        if prompt := st.chat_input(t["chat_input"]):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Generate response
            lang_code = LANGUAGE_MAPPING[output_lang]["code"]
            context = st.session_state.get('patient_context', {
                'age': age,
                'weight': weight,
                'gender': gender,
                'allergies': allergies,
                'conditions': conditions,
                'language': lang_code
            })
            
            # Include prescription context if available
            prescription_context = ""
            if st.session_state.get('analysis_done', False):
                prescription_context = f"""
                **Prescription Context:**
                {st.session_state.prescription_text[:1000]}
                
                **Previous Analysis:**
                {st.session_state.analysis_result[:1000]}
                """
            
            # System prompt for the AI
            system_prompt = f"""
            You are a professional medical assistant named AarogyaAI helping a {context['age']} year old {context['gender']} patient understand their prescription.
            
            **Patient Details:**
            - Weight: {context['weight']} kg
            - Allergies: {', '.join(context['allergies']) or 'None'}
            - Existing Conditions: {', '.join(context['conditions']) or 'None'}
            
            **Instructions:**
            1. Always respond in {output_lang} unless specifically asked to use another language
            2. Focus on the prescription content and patient context
            3. Provide accurate, medically verified information
            4. Explain complex medical terms in simple language
            5. Highlight important warnings clearly
            6. Be empathetic and professional
            7. If unsure about something, say you don't know rather than guessing
            8. Format responses clearly with bullet points when appropriate
            
            {prescription_context}
            """
            
            # Prepare the full prompt for Gemini
            chat_prompt = {
                "system_instruction": system_prompt,
                "parts": [{"text": prompt}]
            }
            
            try:
                with st.spinner("Generating response..."):
                    # Generate response using Gemini 1.5 Flash
                    response = chat_model.generate_content(chat_prompt)
                    answer = response.text
                    
                    # Translate if needed
                    if lang_code != 'en':
                        answer = translate_text(answer, lang_code)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Rerun to display new messages
                    st.rerun()
            
            except Exception as e:
                st.error(f"Chat failed: {str(e)}")

if __name__ == "__main__":
    main()