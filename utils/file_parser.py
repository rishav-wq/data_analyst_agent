import pandas as pd
import docx
import PyPDF2
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract # Requires Tesseract OCR engine installation
import io
import os
import numpy as np

# Additional imports for PDF to image conversion and enhanced OCR
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not installed. OCR for PDFs will be limited.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Info: OpenCV not available. Using basic image preprocessing.")

# Configure Tesseract path if necessary (especially on Windows)
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Uncomment and modify the path below if needed:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def parse_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error parsing CSV: {e}"

def parse_excel(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl') # Specify engine
        return df
    except Exception as e:
        return f"Error parsing Excel: {e}"

def parse_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error parsing DOCX: {e}"

def preprocess_image_for_ocr(image_path):
    """
    Preprocess image to improve OCR accuracy
    """
    try:
        processed_images = []
        
        if OPENCV_AVAILABLE:
            # Use OpenCV for better preprocessing
            img = cv2.imread(image_path)
            if img is None:
                # Fallback to PIL if OpenCV can't read it
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Original grayscale
            processed_images.append(("original_gray", Image.fromarray(gray)))
            
            # Increase contrast and brightness
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
            processed_images.append(("enhanced", Image.fromarray(enhanced)))
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            processed_images.append(("blurred", Image.fromarray(blurred)))
            
            # Threshold (binary)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            processed_images.append(("binary_threshold", Image.fromarray(thresh)))
            
            # Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_images.append(("adaptive_threshold", Image.fromarray(adaptive)))
            
            # OTSU threshold
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("otsu_threshold", Image.fromarray(otsu)))
        
        else:
            # Fallback to PIL-only preprocessing
            img = Image.open(image_path)
            
            # Convert to grayscale
            if img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
            
            processed_images.append(("original", gray_img))
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray_img)
            enhanced_img = enhancer.enhance(2.0)
            processed_images.append(("enhanced_contrast", enhanced_img))
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(gray_img)
            sharp_img = sharpness_enhancer.enhance(2.0)
            processed_images.append(("enhanced_sharpness", sharp_img))
            
            # Apply filter to reduce noise
            filtered_img = gray_img.filter(ImageFilter.MedianFilter())
            processed_images.append(("filtered", filtered_img))
        
        return processed_images
        
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        # Return original image as fallback
        try:
            return [("original", Image.open(image_path))]
        except:
            return []

def parse_image(file_path, debug=False):
    """
    Enhanced image parsing with multiple OCR attempts and preprocessing
    """
    try:
        if debug:
            print(f"Processing image: {file_path}")
            
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found."
        
        # Get image info
        try:
            with Image.open(file_path) as img:
                if debug:
                    print(f"Image size: {img.size}, Mode: {img.mode}, Format: {img.format}")
        except Exception as e:
            if debug:
                print(f"Error reading image info: {e}")
        
        # Test basic Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            if debug:
                print(f"Tesseract version: {version}")
        except Exception as e:
            return f"Tesseract installation error: {e}. Please install Tesseract OCR."
        
        # Try different OCR configurations
        ocr_configs = [
            '--psm 6',  # Uniform block of text
            '--psm 3',  # Default
            '--psm 8',  # Single word
            '--psm 7',  # Single text line  
            '--psm 11', # Sparse text
            '--psm 12', # Sparse text with OSD
            '--psm 13', # Raw line (no word detection)
            '--psm 1',  # Automatic page segmentation with OSD
        ]
        
        best_result = ""
        best_confidence = 0
        
        # Try with different preprocessing
        processed_images = preprocess_image_for_ocr(file_path)
        
        if not processed_images:
            # Final fallback - try with original image
            processed_images = [("fallback", Image.open(file_path))]
        
        for preprocess_name, processed_img in processed_images:
            if debug:
                print(f"\nTrying preprocessing: {preprocess_name}")
            
            for config in ocr_configs:
                try:
                    # Extract text with confidence
                    data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Extract text
                    text = pytesseract.image_to_string(processed_img, config=config).strip()
                    
                    if debug and text:
                        print(f"  Config {config}: confidence={avg_confidence:.1f}, text_length={len(text)}")
                        print(f"    Text preview: {text[:100]}...")
                    
                    # Keep best result based on confidence and text length
                    if text and (avg_confidence > best_confidence or (avg_confidence == best_confidence and len(text) > len(best_result))):
                        best_result = text
                        best_confidence = avg_confidence
                        if debug:
                            print(f"    *** New best result! ***")
                
                except Exception as e:
                    if debug:
                        print(f"  Error with config {config}: {e}")
                    continue
        
        # Try with different languages as fallback
        if not best_result.strip():
            if debug:
                print("\nTrying different languages...")
            
            languages = ['eng', 'eng+osd']  # You can add more languages like 'fra', 'deu', etc.
            
            for lang in languages:
                try:
                    text = pytesseract.image_to_string(Image.open(file_path), lang=lang)
                    if debug:
                        print(f"Language {lang}: {len(text)} characters")
                    if text.strip() and len(text.strip()) > len(best_result.strip()):
                        best_result = text.strip()
                except Exception as e:
                    if debug:
                        print(f"Error with language {lang}: {e}")
        
        # Special handling for UI screenshots
        if not best_result.strip():
            if debug:
                print("\nTrying UI-specific OCR settings...")
            try:
                # For UI screenshots, try with character whitelist
                ui_config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()_+-=[]{}|;:,.<>?/~`'
                text = pytesseract.image_to_string(Image.open(file_path), config=ui_config)
                if text.strip():
                    best_result = text.strip()
                    if debug:
                        print(f"UI-specific OCR found: {len(text)} characters")
            except Exception as e:
                if debug:
                    print(f"Error with UI-specific OCR: {e}")
        
        if best_result.strip():
            if debug:
                print(f"\nFinal result: {len(best_result)} characters, confidence: {best_confidence:.1f}")
            return best_result.strip()
        else:
            return "No text detected in image. Possible issues:\n1. Image quality too low\n2. Text too small or blurry\n3. Complex background interfering with OCR\n4. Unsupported text format\n\nTry using a higher quality image with clear, readable text."
            
    except pytesseract.TesseractNotFoundError:
        return "Error: Tesseract OCR not found. Please install Tesseract and add it to your PATH.\n\nInstallation:\n- Windows: https://github.com/UB-Mannheim/tesseract/wiki\n- Mac: brew install tesseract\n- Linux: sudo apt-get install tesseract-ocr"
    except Exception as e:
        return f"Error parsing image with OCR: {e}"

def perform_ocr_on_pdf(file_path):
    """
    Perform OCR on PDF by converting it to images first
    """
    try:
        if not PDF2IMAGE_AVAILABLE:
            return "Error: pdf2image not installed. Cannot perform OCR on PDF. Install with: pip install pdf2image"
        
        # Convert PDF to images
        pages = pdf2image.convert_from_path(file_path, dpi=200)  # Lower DPI for faster processing
        
        extracted_text = ""
        for page_num, page_image in enumerate(pages, 1):
            try:
                # Use the enhanced image parsing for each page
                # Save page as temporary image
                temp_image_path = f"temp_page_{page_num}.png"
                page_image.save(temp_image_path)
                
                # Use enhanced OCR
                page_text = parse_image(temp_image_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_image_path)
                except:
                    pass
                
                if page_text and not page_text.startswith("Error:") and not page_text.startswith("No text"):
                    extracted_text += f"\n--- Page {page_num} ---\n{page_text.strip()}\n"
                else:
                    extracted_text += f"\n--- Page {page_num} ---\n[No text detected on this page]\n"
            except Exception as e:
                extracted_text += f"\n--- Page {page_num} ---\n[OCR Error: {str(e)}]\n"
        
        return extracted_text.strip() if extracted_text.strip() else "No text could be extracted via OCR."
        
    except Exception as e:
        return f"OCR Error: {str(e)}"

def parse_pdf(file_path):
    try:
        text = ""
        
        # First, try direct text extraction
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        # If no meaningful text was extracted, try OCR
        if not text.strip() or len(text.strip()) < 50:  # Less than 50 chars likely means no real content
            print("PDF appears to be image-based. Attempting OCR...")
            ocr_text = perform_ocr_on_pdf(file_path)
            if ocr_text and not ocr_text.startswith("Error:") and not ocr_text.startswith("OCR Error:"):
                return ocr_text
            else:
                return f"PDF processing failed. Direct extraction: '{text[:100]}...' OCR result: {ocr_text}"
        
        return text.strip()
        
    except Exception as e:
        # If direct extraction fails completely, try OCR as fallback
        print(f"Direct PDF extraction failed: {str(e)}. Attempting OCR...")
        ocr_result = perform_ocr_on_pdf(file_path)
        if ocr_result and not ocr_result.startswith("Error:"):
            return ocr_result
        else:
            return f"Error parsing PDF: {e}. OCR also failed: {ocr_result}"

def parse_file(file_path, debug=False):
    """
    Parses a file based on its extension and returns its content.
    For tabular data (CSV, XLSX), returns a Pandas DataFrame.
    For other types, returns extracted text.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == '.txt':
        return parse_txt(file_path), "text"
    elif extension == '.csv':
        return parse_csv(file_path), "dataframe"
    elif extension == '.xlsx':
        return parse_excel(file_path), "dataframe"
    elif extension == '.docx':
        return parse_docx(file_path), "text"
    elif extension == '.pdf':
        return parse_pdf(file_path), "text"
    elif extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        return parse_image(file_path, debug), "text"
    else:
        return f"Unsupported file type: {extension}", "error"

# Installation requirements check
def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import pytesseract
        # Test if Tesseract is properly configured
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR is properly installed and configured")
    except (ImportError, pytesseract.TesseractNotFoundError):
        missing_deps.append("Tesseract OCR engine")
        print("❌ Tesseract OCR not found or not configured")
    
    if not PDF2IMAGE_AVAILABLE:
        missing_deps.append("pdf2image (pip install pdf2image)")
        print("⚠️ pdf2image not installed")
    else:
        print("✅ pdf2image is available")
    
    if not OPENCV_AVAILABLE:
        print("⚠️ OpenCV not available (recommended for better OCR preprocessing)")
        print("   Install with: pip install opencv-python")
    else:
        print("✅ OpenCV is available for enhanced image preprocessing")
    
    if missing_deps:
        print("\nMissing dependencies for full functionality:")
        for dep in missing_deps:
            print(f"- {dep}")
        print("\nTo install missing Python packages:")
        print("pip install pdf2image opencv-python")
        print("\nTo install Tesseract OCR:")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("- Mac: brew install tesseract")
        print("- Linux: sudo apt-get install tesseract-ocr")
        return False
    return True

def diagnose_ocr_issue(image_path):
    """
    Diagnose OCR issues with a specific image
    """
    print(f"\n=== OCR Diagnosis for {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    try:
        # Check image properties
        with Image.open(image_path) as img:
            print(f"📸 Image Info:")
            print(f"   Size: {img.size}")
            print(f"   Mode: {img.mode}")
            print(f"   Format: {img.format}")
            
            # Check if image is too small
            if img.size[0] < 100 or img.size[1] < 100:
                print("⚠️ Image might be too small for reliable OCR")
            
            # Check if image is grayscale
            if img.mode not in ['L', 'RGB']:
                print(f"⚠️ Unusual image mode: {img.mode}")
    
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return
    
    # Try OCR with debug mode
    result = parse_image(image_path, debug=True)
    print(f"\n📋 Final OCR Result:")
    print(f"Length: {len(result)} characters")
    if result:
        print(f"Preview: {result[:200]}...")
    else:
        print("No text detected")

if __name__ == '__main__':
    # Check dependencies
    print("=== Dependency Check ===")
    check_dependencies()
    
    # Create dummy files for testing
    os.makedirs('uploads', exist_ok=True)
    with open('uploads/test.txt', 'w') as f:
        f.write("This is a test text file.")
    
    data = {'col1': [1, 2], 'col2': ['a', 'b']}
    df_csv = pd.DataFrame(data)
    df_csv.to_csv('uploads/test.csv', index=False)

    # Writing to Excel:
    with pd.ExcelWriter('uploads/test.xlsx', engine='openpyxl') as writer:
        df_csv.to_excel(writer, sheet_name='Sheet1', index=False)

    print("\n=== Testing File Parsing ===")
    
    content, content_type = parse_file('uploads/test.txt')
    print(f"TXT Content ({content_type}):\n{content}\n")

    content, content_type = parse_file('uploads/test.csv')
    if content_type == "dataframe":
        print(f"CSV Content ({content_type}):\n{content.head()}\n")
    else:
        print(f"CSV Content ({content_type}):\n{content}\n")

    content, content_type = parse_file('uploads/test.xlsx')
    if content_type == "dataframe":
        print(f"XLSX Content ({content_type}):\n{content.head()}\n")
    else:
        print(f"XLSX Content ({content_type}):\n{content}\n")
    
    # If you have a test image, uncomment the line below and replace with your image path
    # diagnose_ocr_issue('path/to/your/test_image.png')