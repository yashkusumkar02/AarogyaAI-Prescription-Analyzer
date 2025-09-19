
## 🏥 AarogyaAI - Multilingual Prescription Analyzer


AarogyaAI is an AI-powered system that interprets medical prescriptions written in multiple languages using Google Gemini 1.5 Flash and OCR. It flags potential drug interactions, dosage errors, and allergies — all through a simple and intuitive interface.

<img width="1918" height="910" alt="image" src="https://github.com/user-attachments/assets/16872980-af89-451a-9743-e6f3f50e99f4" />


🚀 Features

- 📄 Multilingual OCR: Extracts prescription text from images in English, Hindi, and more.
- ⚠️ Medication Insights: Identifies drug interactions, dosage issues, and allergy alerts.
- 🌍 User-Friendly UI: Built with Streamlit, allowing users to upload images and see highlighted medical insights.
- 🎯 High Accuracy: Achieves ~95% text extraction accuracy via optimized image preprocessing.

🛠️ Installation

1. Clone the repository:
   git clone https://github.com/your-username/AarogyaAI-Prescription-Analyzer.git
   cd AarogyaAI-Prescription-Analyzer

2. Install dependencies:
   pip install -r requirements.txt

3. Set up your API keys:

   - Rename .env.example to .env
   - Add your Gemini API key:
     GEMINI_API_KEY=your_api_key_here

4. Run the app:
   streamlit run app/main.py

🧪 Usage

1. Upload a JPEG, PNG, or PDF of the prescription.
2. Select the language of the document.
3. View the extracted text and detailed medication analysis.
4. Warnings (if any) will be clearly highlighted.

🧠 Tech Stack

- AI/ML: Gemini 1.5 Flash, LangChain  
- OCR: PyTesseract, OpenCV  
- Backend: Python, Pandas  
- Frontend: Streamlit  
- Deployment: Streamlit Cloud / Hugging Face Spaces (Optional)

📁 Project Structure

AarogyaAI/
├── app/                  # Streamlit application (main UI)
│   └── main.py
├── utils/                # Helper scripts
│   ├── ocr.py            # Text extraction logic
│   └── gemini_handler.py # AI analysis module
├── assets/               # Sample prescriptions/test images
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
└── README.md             # You are here!

📄 License

Distributed under the MIT License. See LICENSE for details.

📬 Contact

For questions or collaborations, reach out to:

Suyash Prakash Kusumkar: kusumkarsuyash1234@gmail.com 
GitHub: @yashkusumkar02  
LinkedIn: https://www.linkedin.com/in/suyash-kusumkar/
