import json
import os

def standardize_datasets(file_list):
    formatted_files = []
    
    for file_name in file_list:
        if not os.path.exists(file_name):
            print(f"⚠️ Không tìm thấy file: {file_name}")
            continue
            
        print(f"🔄 Đang xử lý: {file_name}...")
        standardized_data = []
        
        with open(file_name, 'r', encoding='utf-8') as f:
            # Xử lý cả JSON chuẩn ([...]) và JSONL (từng dòng)
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]

        for item in data:
            # Logic quan trọng: Chuyển question -> query
            if 'question' in item:
                item['query'] = item.pop('question')
            
            # Đảm bảo các field tối thiểu của class Instance tồn tại
            item.setdefault('jailbreak_prompt', '')
            item.setdefault('target_responses', [])
            item.setdefault('eval_results', [])
            
            standardized_data.append(item)

        # Lưu file mới để không ghi đè lên file gốc
        new_name = f"formatted_{file_name}"
        if not new_name.endswith('.jsonl'):
            new_name = new_name.rsplit('.', 1)[0] + '.jsonl'
            
        with open(new_name, 'w', encoding='utf-8') as f:
            for entry in standardized_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        formatted_files.append(new_name)
        print(f"✅ Đã xong! Lưu tại: {new_name}")
        
    return formatted_files

# --- THỰC THI ---
file_names = [
    'AdversarialQA.json', 'BeaverTails.json', 'CategoricalHarmfulQA.json', 
    'DangerousQA.json', 'do-not-answer.json', 'harmbench.json', 'harmfulQA.json'
]

new_files = standardize_datasets(file_names)