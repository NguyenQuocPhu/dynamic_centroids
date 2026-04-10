import json
import os

def standardize_to_json(file_list):
    formatted_files = []
    
    for file_name in file_list:
        if not os.path.exists(file_name):
            print(f"⚠️ Không tìm thấy file: {file_name}")
            continue
            
        print(f"🔄 Đang xử lý: {file_name}...")
        standardized_data = []
        
        # 1. Đọc dữ liệu (Hỗ trợ cả JSON và JSONL đầu vào)
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                continue
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]

        # 2. Chuẩn hóa dữ liệu
        for item in data:
            if 'question' in item:
                item['query'] = item.pop('question')
            

            standardized_data.append(item)

        # 3. Lưu file dưới dạng JSON chuẩn (List of dicts)
        # Đổi đuôi file thành .json
        new_name = f"formatted_{file_name.rsplit('.', 1)[0]}.json"
            
        with open(new_name, 'w', encoding='utf-8') as f:
            # Lưu toàn bộ list vào một file JSON duy nhất, có thụt đầu dòng (indent) cho đẹp
            json.dump(standardized_data, f, ensure_ascii=False, indent=4)
        
        formatted_files.append(new_name)
        print(f"✅ Đã xong! File JSON chuẩn tại: {new_name}")
        
    return formatted_files

# --- THỰC THI ---
file_names = [
    'AdversarialQA.json', 'BeaverTails.json', 'CategoricalHarmfulQA.json', 
    'DangerousQA.json', 'do-not-answer.json', 'harmbench.json', 'harmfulQA.json'
]

new_files = standardize_to_json(file_names)