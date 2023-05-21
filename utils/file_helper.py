class FileHelper:

 def open_file(self,filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


 def save_file(self,filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)