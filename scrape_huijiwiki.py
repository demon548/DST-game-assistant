"""
灰机Wiki爬虫 — 从 dontstarve.huijiwiki.com 补充知识库
使用MediaWiki API获取页面内容
"""
import cloudscraper
from bs4 import BeautifulSoup
import os, time, re, json, urllib.parse

scraper = cloudscraper.create_scraper()
OUTPUT = "data/raw"
API = "https://dontstarve.huijiwiki.com/api.php"

# URL编码后的页面名 (直接从首页链接提取)
PAGES = [
    # === 游戏机制 ===
    ("游戏机制", "%E6%9C%88%E7%9B%B8", "月相.txt"),
    ("游戏机制", "%E6%98%BC%E5%A4%9C%E5%BE%AA%E7%8E%AF", "昼夜循环.txt"),
    ("游戏机制", "%E5%86%AC%E5%AD%A3", "冬季.txt"),
    ("游戏机制", "%E5%A4%8F%E5%AD%A3", "夏季.txt"),
    ("游戏机制", "%E6%98%A5%E5%AD%A3", "春季.txt"),
    ("游戏机制", "%E5%85%89%E6%BA%90", "光源.txt"),
    ("游戏机制", "%E5%86%B0%E5%86%BB", "冰冻.txt"),
    ("游戏机制", "%E5%82%AC%E7%9C%A0", "催眠.txt"),
    ("游戏机制", "%E5%9C%B0%E9%9C%87", "地震.txt"),
    ("游戏机制", "%E5%B9%B8%E8%BF%90%E5%80%BC", "幸运值.txt"),
    ("游戏机制", "%E5%90%AF%E8%92%99%E5%80%BC", "启示值.txt"),
    ("游戏机制", "%E4%BC%A4%E5%AE%B3", "伤害.txt"),
    ("游戏机制", "%E4%BD%8D%E9%9D%A2%E4%BC%A4%E5%AE%B3", "位面伤害.txt"),
    ("游戏机制", "%E6%9A%B4%E9%A3%9F", "暴食.txt"),
    ("游戏机制", "%E6%8E%A7%E5%88%B6%E5%8F%B0", "控制台.txt"),

    # === 世界 ===
    ("世界", "%E4%B8%96%E7%95%8C%E5%86%8D%E7%94%9F", "世界再生.txt"),
    ("世界", "%E4%B8%96%E7%95%8C%E7%BF%BB%E6%96%B0", "世界翻新.txt"),
    ("世界", "%E5%9C%B0%E5%9D%97", "地块.txt"),
    ("世界", "%E5%8C%BA%E5%9D%97", "区块.txt"),

    # === 事件 ===
    ("事件", "%E6%9A%97%E5%BD%B1%E8%A3%82%E9%9A%99%E5%BE%AA%E7%8E%AF", "暗影裂隙循环.txt"),
    ("事件", "%E6%9C%88%E4%BA%AE%E8%A3%82%E9%9A%99%E5%BE%AA%E7%8E%AF", "月亮裂隙循环.txt"),
    ("事件", "%E6%9C%88%E4%BA%AE%E9%A3%8E%E6%9A%B4", "月亮风暴.txt"),
    ("事件", "%E6%A2%A6%E9%AD%87%E5%BE%AA%E7%8E%AF", "梦魇循环.txt"),
    ("事件", "%E6%9C%88%E9%9B%B9", "月雹.txt"),
    ("事件", "%E6%9C%88%E4%BA%AE%E5%8F%98%E5%BC%82", "月亮变异.txt"),

    # === 角色 ===
    ("角色", "%E5%A8%81%E5%B0%94%E9%80%8A", "威尔逊.txt"),
    ("角色", "%E6%97%BA%E8%BE%BE", "旺达.txt"),

    # === 物品/建筑 ===
    ("物品", "%E6%9A%97%E5%A4%9C%E7%81%AF", "暗夜灯.txt"),
    ("物品", "%E6%87%92%E4%BA%BA%E9%AD%94%E6%9D%96", "懒人魔杖.txt"),
    ("物品", "%E5%9C%B0%E7%83%AD%E8%9E%A8", "地热螨.txt"),

    # === 更新 ===
    ("更新", "%E5%92%92%E7%BB%88%E5%AF%B9%E5%86%B3%C2%B7%E4%B8%8A%E9%83%A8", "咒终对决上部.txt"),
    ("更新", "%E5%9C%A3%E6%AE%BF", "圣殿.txt"),
]

# 也用中文名做API搜索，获取更精确的结果
SEARCH_TERMS = {
    "月相": "游戏机制",
    "月亮周期": "游戏机制",
    "温度": "游戏机制",
    "雨": "游戏机制",
    "理智": "游戏机制",
    "食物度": "游戏机制",
    "生命值": "游戏机制",
    "走位": "游戏机制",
    "烹饪锅": "食物",
    "耕种": "食物",
    "食谱": "食物",
    "农田": "食物",
    "四季": "四季",
    "秋天": "四季",
    "蜘蛛": "生物",
    "猪人": "生物",
    "兔人": "生物",
    "猎犬": "生物",
    "高脚鸟": "生物",
    "巨鹿": "Boss",
    "熊獾": "Boss",
    "龙蝇": "Boss",
    "蜂后": "Boss",
    "克劳斯": "Boss",
    "织影者": "Boss",
    "帝王蟹": "Boss",
    "天体英雄": "Boss",
    "威尔逊": "角色",
    "温蒂": "角色",
    "沃尔夫冈": "角色",
    "韦伯": "角色",
    "沃利": "角色",
    "沃拓克斯": "角色",
    "沃姆伍德": "角色",
    "麦斯威尔": "角色",
    "WX-78": "角色",
    "武器": "装备",
    "工具": "装备",
    "护甲": "装备",
    "衣物": "装备",
    "合成": "制作",
    "新手": "指南",
    "生存": "指南",
    "战斗": "指南",
    "韦斯": "角色",
    "薇洛": "角色",
    "沃特": "角色",
    "薇克伯顿": "角色",
    "薇诺娜": "角色",
}


def get_page_text(page_id: str) -> str:
    """通过API获取页面文本内容"""
    params = {
        'action': 'parse',
        'page': page_id,
        'prop': 'text',
        'format': 'json'
    }
    try:
        r = scraper.get(API, params=params, timeout=20)
        data = r.json()
        if 'parse' in data:
            html = data['parse']['text']['*']
            soup = BeautifulSoup(html, 'lxml')
            # 移除不需要的元素
            for tag in soup.find_all(['table', 'script', 'style', 'sup', 'nav']):
                tag.decompose()
            for cls in ['toc', 'navbox', 'mw-editsection', 'thumbcaption',
                        'sprite']:
                for div in soup.find_all('div', class_=re.compile(cls)):
                    div.decompose()

            text = soup.get_text('\n')
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 2]
            return '\n'.join(lines)
        return ""
    except Exception as e:
        print(f"      错误: {e}")
        return ""


def search_and_get(query: str) -> str:
    """搜索页面并获取内容"""
    # 先搜索
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json'
    }
    try:
        r = scraper.get(API, params=params, timeout=20)
        data = r.json()
        results = data.get('query', {}).get('search', [])
        if results:
            title = results[0]['title']
            return get_page_text(title)
    except Exception as e:
        print(f"      搜索错误: {e}")
    return ""


def main():
    # 先清掉之前的测试输出
    test_dir = os.path.join(OUTPUT, "灰机Wiki")
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)

    total_chars = 0
    success = 0

    print("=" * 60)
    print("灰机Wiki 知识库补充 — 使用MediaWiki API")
    print("=" * 60)

    # 第一步: 爬取URL编码的已知页面
    print("\n--- 阶段1: 已知页面 ---")
    for i, (category, encoded_name, filename) in enumerate(PAGES):
        print(f"[{i+1}/{len(PAGES)}] {category} → {filename}")
        text = get_page_text(encoded_name)

        if text and len(text) > 300:
            cat_dir = os.path.join(OUTPUT, "灰机Wiki", category)
            os.makedirs(cat_dir, exist_ok=True)
            filepath = os.path.join(cat_dir, filename)
            decoded = urllib.parse.unquote(encoded_name)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {decoded}\n来源: https://dontstarve.huijiwiki.com/wiki/{encoded_name}\n\n{text}")

            chars = len(text)
            total_chars += chars
            success += 1
            print(f"  ✅ {chars:,} 字符")
        else:
            print(f"  ❌ 内容不足")

        time.sleep(1)

    # 第二步: 搜索补充其他主题
    print(f"\n--- 阶段2: 搜索补充 ({len(SEARCH_TERMS)}个主题) ---")
    for term, category in SEARCH_TERMS.items():
        print(f"搜索: {term}")
        text = search_and_get(term)

        if text and len(text) > 300:
            cat_dir = os.path.join(OUTPUT, "灰机Wiki", category)
            os.makedirs(cat_dir, exist_ok=True)
            filename = f"{term}.txt"
            filepath = os.path.join(cat_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {term}\n来源: 灰机Wiki搜索\n\n{text}")

            chars = len(text)
            total_chars += chars
            success += 1
            print(f"  ✅ {chars:,} 字符")
        else:
            print(f"  ❌ 未找到或内容不足")

        time.sleep(1.5)

    print("\n" + "=" * 60)
    print(f"爬取完成!")
    print(f"  成功: {success} 个页面")
    print(f"  总字符: {total_chars:,} (~{total_chars//1500} 页)")
    print(f"  文件保存在: {OUTPUT}/灰机Wiki/")


if __name__ == "__main__":
    main()
