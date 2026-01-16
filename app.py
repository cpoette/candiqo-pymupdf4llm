from flask import Flask, request, jsonify
import pymupdf4llm
import os
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "pdf-extractor"})

@app.route('/extract', methods=['POST'])
def extract():
    """Extract text from PDF - accepts both multipart and JSON+base64"""
    
    temp_path = None  # Init pour le finally
    
    try:
        # Try multipart first
        if 'pdf' in request.files:
            pdf_file = request.files['pdf']
            temp_path = f"/tmp/{pdf_file.filename}"
            pdf_file.save(temp_path)
        
        # Try JSON with base64
        elif request.is_json:
            data = request.get_json()
            if 'pdf_base64' not in data:
                return jsonify({"error": "No PDF data"}), 400
            
            pdf_base64 = data['pdf_base64']
            filename = data.get('filename', 'document.pdf')
            
            # Decode base64
            try:
                pdf_bytes = base64.b64decode(pdf_base64)
            except Exception as e:
                return jsonify({"error": f"Invalid base64: {str(e)}"}), 400
            
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(pdf_bytes)
        
        else:
            return jsonify({"error": "No PDF provided"}), 400
        
        # Log avant extraction
        logger.info(f"Fichier sauvegardé: {temp_path}")
        logger.info(f"Taille fichier: {os.path.getsize(temp_path)} bytes")
        
        # EXTRACTION
        markdown_text = pymupdf4llm.to_markdown(
            temp_path,
            page_chunks=False,
            write_images=False
        )
        
        # DEBUG LOGS
        logger.info(f"✓ Extraction terminée")
        logger.info(f"  - Caractères extraits: {len(markdown_text)}")
        logger.info(f"  - Est vide: {len(markdown_text) == 0}")
        logger.info(f"  - Premiers 200 chars: {markdown_text[:200]}")
        
        return jsonify({
            "success": True,
            "markdown": markdown_text,
            "extraction_method": "pymupdf4llm",
            "char_count": len(markdown_text),
            "has_content": len(markdown_text) > 0
        })
    
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    
    finally:
        # Cleanup dans tous les cas
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Fichier temporaire supprimé: {temp_path}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)