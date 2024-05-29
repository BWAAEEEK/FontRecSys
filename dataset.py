import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ViTImageProcessor


font_class = {'Ultra': 0, 'WorkSans Bold': 1, '지마켓 산스 B': 2, '지마켓 산스 M': 3, 'Roundo Bold': 4, '나눔스퀘어 R': 5
                , '8비트 B': 6, 'Playfair Display R': 7, '수명조 20': 8, 'Playfair Display RI': 9, '나눔스퀘어 B': 10
                , 'DM Serif Display I': 11, '프리텐다드 Light': 12, '나눔스퀘어 EB': 13, '프리텐다드 ExtraBold': 14
                , '프리텐다드 SemiBold': 15, '프리텐다드 Regular': 16, '프리텐다드 Bold': 17, 'TeXGyreAdventor': 18
                , 'Aileron Black': 19, '조선일보 명조': 20, 'VarelaRound': 21, 'BEBASNEUE': 22, 'WorkSans Regular': 23
                , '프리텐다드 Black': 24, '굴림 070': 25, 'FredokaOne': 26, '제주고딕': 27, '노토 산스 Bold': 28
                , '노토 산스 Medium': 29, '노토 산스 Black': 30, '노토 산스 Regular': 31, 'Bungee': 32, 'Poppins Light': 33
                , '나눔스퀘어 네오 Heavy': 34, '나눔스퀘어 네오 EB': 35, 'Fjalla One': 36, 'Allura': 37, '나눔스퀘어 L': 38
                , 'Italiana': 39, 'Aleo Light': 40, '수명조 10': 41, 'Playfair Display SB': 42, 'Playfair Display M': 43, '화명조 030': 44, 'Poppins Bold': 45, 'Librebaskerville Regular': 46, '부크크 명조 L': 47, '수명조 40': 48, 'Playfair Display BI': 49, '격동고딕': 50, '나눔손글씨 소미체': 51, '두넌': 52, 'Roboto': 53, '라인씨드 B': 54, '라인씨드 R': 55, '롯데리아 촵땡겨체': 56, 'Fugaz One': 57, '잘난': 58, 'MPLUS1p Bold': 59, '비밀정원 B': 60, 'Prompt Regular': 61, '옴니고딕 040': 62, 'Prompt Medium': 63, 'Gravitas One': 64, '옴니고딕 050': 65, '116 앵무부리': 66, 'Leckerli One': 67, 'OpenSans Regular': 68, 'BEBASNEUE BOOK': 69, 'Marcellus SC': 70, '오아 고딕 M': 71, '오아 고딕 EB': 72, '나눔스퀘어 네오 L': 73, 'Montserrat Thin': 74, '수트 Regular': 75, 'Montserrat Regular': 76, '수트 Medium': 77, 'Prompt Light': 78, 'Playfair Display B': 79, 'Kopub바탕 M': 80, 'Kopub바탕 L': 81, '별빛 B': 82, 'Creepster': 83, '국민체조 B': 84, '국민체조 L': 85, '8비트 R': 86, 'ONE 모바일고딕 Bold': 87, 'ONE 모바일고딕 Regular': 88, '자연공원 B': 89, '자연공원 L': 90, 'Prompt ExtraBold': 91, '네모니2 Bd': 92, '더 잠실 M': 93, '넥슨 Lv.2 고딕 R': 94, '독립돋움 R': 95, '넥슨 Lv.2 고딕 M': 96, '프리텐다드 ExtraLight': 97, '고도B': 98, 'Raleway ExtraBold': 99, 'Tlab 더클래식': 100, '이서윤체': 101, '읏맨 궁서체': 102, 'SB 어그로 L': 103, '평화 R': 104, '영도체': 105, '에스코어드림 9': 106, '에스코어드림 4': 107, '나눔명조 B': 108, '비밀정원 R': 109, '햄릿 Light': 110, '햄릿 Medium': 111, '공게임즈 이사만루체 M': 112, '교차로플러스 R': 113, '공게임즈 이사만루체 B': 114, 'Domine-Regular': 115, 'KBIZ 한마음명조 R': 116, 'KBIZ 한마음명조 M': 117, 'KBIZ 한마음명조 B': 118, '수퍼사이즈BlackBOX': 119, 'Noto Sans Bold': 120, 'Noto Sans Bold Italic': 121, 'Noto Sans Regular': 122, '수트 Light': 123, 'Chivo Regular': 124, 'Calistoga': 125, 'Diplomata': 126, '나눔손글씨 강인한위로': 127, 'Kopub바탕 B': 128, '나눔손글씨 손편지체': 129, '더 잠실 L': 130, '더 잠실 R': 131, '아이돌고딕 R': 132, 'KBIZ 한마음고딕 B': 133, 'KBIZ 한마음고딕 M': 134, 'KBIZ 한마음고딕 R': 135, 'JTT 응급': 136, 'SB 어그로 B': 137, '굴림 090': 138, '정묵바위': 139, '나눔명조 EB': 140, '오잉 twinkle': 141, 'JTT 참잘했어요': 142, 'Changa One': 143, 'Cinzel Decorative': 144, '디딤명조 030': 145, '디딤명조 020': 146, 'Homemade Apple': 147, '나눔스퀘어라운드 R': 148, '나눔스퀘어라운드 L': 149, '디딤명조 010': 150, '나무굴림 EB': 151, 'HS산토끼': 152, '별빛차 R': 153, '한국기계연구원 B': 154, 'JTT 알쏭달쏭 B': 155, 'HS새마을체': 156, 'IM 혜민 B': 157, '프리텐다드 Thin': 158, '넥슨 Lv.2 고딕 L': 159, 'Mrs Saint Delafield': 160, '개성시대 R': 161, '개성시대 L': 162, '마마블럭 R': 163, '나눔스퀘어라운드EB': 164, '나눔스퀘어라운드 B': 165, '락 Sans Bold': 166, '광안리 H': 167, 'Merriweather Black': 168, 'PantonCaps Black': 169, 'JosefinSans': 170, '이누아리두리': 171, 'Droid Sans': 172, 'Condiment': 173, '에스코어드림 3': 174, '데이라잇 R': 175, 'DM Serif Display R': 176, '예스명조 B': 177, '조선 굵은 명조': 178, '예스명조 R': 179, '밀양 아리랑체': 180, '화명조 040': 181, '화명조 020': 182, '액션스텐실 Solid': 183, '엘리스디지털배움체 B': 184, '엘리스디지털배움체 R': 185, '옴니고딕 030': 186, '햄릿 Bold': 187, '오늘은 B': 188, '경기천년바탕 Bold': 189, '상주 경천섬체': 190, 'Rye': 191, 'Barriecito': 192, 'LeagueGothic': 193, '쿠키런 Black': 194, '누리고딕 B': 195, '누리고딕 R': 196, '오동통 Basic': 197, 'Baumans': 198, '상주 다정다감체': 199, '수트 ExtraBold': 200, 'ONE 모바일고딕 Title': 201, '코코 Tree': 202, 'TenorSans': 203, '신문명조 B': 204, 'Lobster R': 205, 'Shrikhand': 206, 'Prompt Black': 207, 'Prata': 208, '햄릿 SemiBold': 209, '석보상절 B': 210, '월인석보': 211, 'Great Vibes': 212, '경기천년바탕': 213, '수필명조 L': 214, '한컴 산스 L': 215, '한컴 산스 SB': 216, '수업시간 L': 217, '수업시간 B': 218, '네온사인 Bold': 219, '아일랜드 R': 220, '123RF': 221, '락 Sans Regular': 222, 'Black Ops One': 223, '옴니고딕 Cond 040': 224, 'JTT 꼬마김밥 R': 225, '몬드리안 Reverse': 226, '누리고딕 L': 227, 'Ma Shan Zheng': 228, '공병각펜': 229, '에스코어드림 8': 230, '수퍼사이즈Black': 231, '쿠키런 Bold': 232, '트리거 R': 233, 'Comfortaa': 234, '강원교육모두 B': 235, 'Orbitron': 236, '옴니굴림 50': 237, '수업시간 R': 238, '역전다방 R': 239, 'AlexBrush': 240, 'Mapo애민': 241, '여씨향약언해 R': 242, '독립명조 R': 243, 'EBGaramond ExtraBold': 244, '넥슨 Lv.1 고딕 B': 245, '에스코어드림 2': 246, 'Roundo Medium': 247, 'Roundo ExtraLight': 248, '옴니고딕 010': 249, 'Mali Light': 250, '캐모마일 L': 251, '캐모마일 B': 252, '배달의민족 주아': 253, '마녀의 숲 B': 254, '마녀의 숲 R': 255, '나눔고딕 R': 256, 'Droid Serif': 257, '나눔바른고딕 R': 258, '나눔바른고딕 L': 259, 'Lato Regular': 260, '아일랜드 B': 261, '아일랜드 L': 262, '나눔바른펜 R': 263, '발레리노': 264, '자유로 Black': 265, '카페24 빛나는별': 266}

align_class = {'center': 0, 'right': 1, 'left': 2, 'justify': 3}


class FeatureDataset(Dataset):
    def __init__(self, data, image_data, node_idx):
        super(FeatureDataset, self).__init__()
        self.data = data
        self.image_data = image_data
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.image_processor = ViTImageProcessor.from_pretrained('facebook/sam-vit-base')
        self.node_idx = node_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):  # 여기서 item = index 라 할 수 있음
        item = self.data[item]

        ppt_file_name = item["ppt_data_file_name"][:-4]
        slide_num = item["slide_num"]

        image = self.image_data[ppt_file_name][slide_num - 1]

        transform = item['transform']
        opacity = item['opacity']
        text_align = align_class[item['text-align']]
        text_align = align_class[item['text-align']]
        line_height = item['line-height']
        letter_spacing = item['letter-spacing']
        font_size = item['font-size']
        font = -1

        bold = int(item['bold'])
        italic = int(item['italic'])
        under_line = int(item['under_line'])
        line = int(item['line'])
        uppercase = int(item['uppercase'])
        text_join = ''

        label = font_class[item['font']]

        for i in item['text']:
            text_join += i
        text = self.tokenizer(text_join, padding='max_length', truncation=True, max_length=100, return_tensors='pt')

        # (1, 500) -> (500)으로 shape 변경
        text = {k: v.squeeze(0) for k, v in text.items()}

        result = torch.FloatTensor([transform, opacity, text_align, line_height, letter_spacing, font_size, font, bold, italic, under_line, line, uppercase])

        return {'text': text, 'result': result, "image": image, "ppt_node": self.node_idx[ppt_file_name], "label": label}


class FinalDataset(Dataset):
    def __init__(self, feature_dataset, image_data, node_idx):
        super(FinalDataset, self).__init__()
        self.feature_dataset = feature_dataset
        self.image_data = image_data
        self.node_idx = node_idx

    def __len__(self):
        return len(self.feature_dataset)

    def __getitem__(self, idx):
        feature_data = self.feature_dataset[idx]
        image_data = self.image_data[idx]




