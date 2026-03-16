import os
import requests # thư viện để gửi request đến server
from bs4 import BeautifulSoup # thư viện để phân tích HTML
from datetime import datetime

current_datetime = datetime.now()
# định dạng thời gian hiện tại thành chuỗi, để tạo tên thư mục lưu trữ các bài viết
folder_name = current_datetime.strftime('%H_%M_%d_%m_%Y')  


# tạo thư mục chính để lưu trữ các folder_name
main_directory = "Saved_Articles"

# tạo thư mục chính nếu nó không tồn tại
if not os.path.exists(main_directory):
    os.makedirs(main_directory, exist_ok=True)

# tạo thư mục con trong thư mục chính, với tên là folder_name
subdirectory_path = os.path.join(main_directory, folder_name)
if not os.path.exists(subdirectory_path):
    os.makedirs(subdirectory_path, exist_ok=True)

# định nghĩa URL để lấy dữ liệu
url = "https://content.guardianapis.com/technology/artificialintelligenceai?&api-key=01dd6b39-66d5-4ed8-8335-9dd17fe41a3f&type=article&page=1"

response = requests.get(url)  # gửi request đến URL và lấy dữ liệu
x = response.json()  # chuyển đổi dữ liệu thành định dạng JSON

web_urls = [item['webUrl'] for item in x['response']['results']] # lấy URL của các bài viết chi tiết

def save_content_to_file(url, folder, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # BeautifulSoup là thư viện để phân tích HTML, để lấy các thẻ h1 và p
            soup = BeautifulSoup(response.text, 'html.parser')

            with open(os.path.join(folder, filename), 'w', encoding='utf-8') as file:
                for header in soup.find_all(['h1']):
                    file.write("Title: " + header.text + '\n' * 5)
                for paragraph in soup.find_all('p'):
                    file.write(paragraph.text + '\n')
        else:
            print("Failed to retrieve the page:", url)
    except Exception as e:
        print("An error occurred:", e)

for index, url in enumerate(web_urls):
    filename = f'article_{index}.txt'  # tạo tên file lưu trữ các bài viết
    save_content_to_file(url, subdirectory_path, filename)  # lưu nội dung của các bài viết vào file trong thư mục con