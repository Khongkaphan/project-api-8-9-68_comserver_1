import re

text = "01 ก.ย. 2568"
pattern = r"\d{2}\s(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s\d{4}"


match = re.search(pattern, text)
print(match)
