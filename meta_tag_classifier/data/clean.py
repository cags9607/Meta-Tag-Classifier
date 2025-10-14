import re, unicodedata, html
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Iterable, Optional

# ---------- base regex & helpers ----------
WS = re.compile(r"\s+")
WORD_SPLIT = re.compile(r"\s+")
ZERO_WIDTH = "".join(["\u200b","\u200c","\u200d","\ufeff"])
BIDI = "".join(["\u202a","\u202b","\u202d","\u202e","\u202c"])

RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S|re.I)
RE_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.S|re.I)
RE_STYLE_BLOCK  = re.compile(r"<style\b[^>]*>.*?</style\s*>",  re.S|re.I)
RE_NOSCRIPT_BLK = re.compile(r"<noscript\b[^>]*>.*?</noscript\s*>", re.S|re.I)
RE_TAGS         = re.compile(r"<[^>]+>", re.S)
RE_TEMPLATES    = re.compile(r"(\{\{.*?\}\}|\{%.*?%}|\<\?.*?\?>)", re.S)

def strip_html_js(s: str) -> str:
    if not isinstance(s, str) or not s.strip(): return ""
    s = html.unescape(s)
    s = RE_HTML_COMMENT.sub(" ", s)
    s = RE_SCRIPT_BLOCK.sub(" ", s)
    s = RE_STYLE_BLOCK.sub(" ", s)
    s = RE_NOSCRIPT_BLK.sub(" ", s)
    s = RE_TAGS.sub(" ", s)
    s = RE_TEMPLATES.sub(" ", s)
    return WS.sub(" ", s).strip()

SEP_RUNS = re.compile(r"[|\-–—•·:]{2,}")
def nfkc(s: str) -> str:
    if not isinstance(s, str): return ""
    if s.strip().lower() in {"nan","none"}: return ""
    s = s.replace("\u00A0"," ").replace("\u200B"," ")
    s = unicodedata.normalize("NFKC", s)
    s = SEP_RUNS.sub("|", s)
    return WS.sub(" ", s).strip()

EMOJI = re.compile(r"[\U0001F1E0-\U0001F1FF\U0001F300-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF]")
EMOTICON = re.compile(r"\b(?:[:;=]-?[\)\]\(\[dDpP/:\}\{@\|\\]|[:;=][_']?[\)\]\(\[dDpP/:\}\{@\|\\]|<3)\b")
def collapse_emotes(s: str) -> str:
    if not s: return s
    s = EMOJI.sub(" ", s)
    s = EMOTICON.sub(" ", s)
    s = re.sub(r"(?:\s*\[EMOJI\]\s*){2,}", " [EMOJI] ", s)
    return WS.sub(" ", s).strip()

URL   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE = re.compile(r"(?:(?<!\d)\+?\d[\d\-\s()]{6,}\d(?!\d))")
DOMAIN = re.compile(r"(?<!@)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:xn--[a-z0-9-]{2,63}|[a-z]{2,63})\b", re.I)
def _domain_repl(_m): return " "

def mask_tokens_smart(s: str) -> str:
    s = URL.sub(" ", s)
    s = EMAIL.sub(" ", s)
    s = PHONE.sub(" ", s)
    s = DOMAIN.sub(_domain_repl, s)
    return WS.sub(" ", s).strip()

def clean_unit(s: str) -> str:
    return mask_tokens_smart(collapse_emotes(nfkc(strip_html_js(s or ""))))

def collapse_word_runs(s: str) -> str:
    toks = WORD_SPLIT.split(s.strip()) if s else []
    out, prev = [], None
    for t in toks:
        if t == prev: continue
        out.append(t); prev = t
    return " ".join(out)

def collapse_ngram_runs(s: str, n: int = 2) -> str:
    toks = WORD_SPLIT.split(s.strip()) if s else []
    out, i, L = [], 0, len(toks)
    while i < L:
        out.extend(toks[i:i+n]); j = i + n
        while j + n <= L and toks[i:i+n] == toks[j:j+n]: j += n
        i = j
    return " ".join(out)

def dedupe_inside(s: str) -> str:
    if not s: return s
    s = collapse_word_runs(s)
    s = collapse_ngram_runs(s, 2)
    s = collapse_ngram_runs(s, 3)
    s = re.sub(r"([!?。！，,\.])\1{1,}", r"\1", s)
    return WS.sub(" ", s).strip()

def clean_text_for_distil(text: str) -> str:
    cleaned = dedupe_inside(clean_unit(text))
    cleaned = re.sub(r'[^\w\s.,!?;:]', '', cleaned)
    cleaned = re.sub(r'(?<!\w)\W(?!\w)', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()

def check_domain_in_text(domain: str, text: str) -> int:
    if not isinstance(domain, str) or not isinstance(text, str): return 0
    parts = domain.split('.'); domain_word = parts[-2] if len(parts) > 1 else parts[0]
    pattern = r'\b' + re.escape(domain_word) + r'\b'
    return 1 if re.search(pattern, text, re.IGNORECASE) else 0

def remove_domain_from_text(domain: str, text: str, has_domain_name: int) -> str:
    if has_domain_name == 1 and isinstance(domain, str) and isinstance(text, str):
        parts = domain.split('.'); domain_word = parts[-2] if len(parts) > 1 else parts[0]
        cleaned = re.sub(r'\b' + re.escape(domain_word) + r'\b', '', text, flags=re.IGNORECASE).strip()
        return re.sub(r'\s+', ' ', cleaned)
    return text

def jaccard_similarity(text1: str, text2: str) -> float:
    if not isinstance(text1, str) or not isinstance(text2, str): return 0.0
    w1, w2 = set(text1.lower().split()), set(text2.lower().split())
    if not w1 and not w2: return 1.0
    if not w1 or not w2:  return 0.0
    return len(w1 & w2) / len(w1 | w2)

def nfkc_casefold(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\x00","")
    s = s.translate({ord(c): None for c in ZERO_WIDTH + BIDI})
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()

def tokenize_basic(s: str):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold())

DEFAULT_NAV_PHRASES = sorted(set([
    "home","menu","search","subscribe","sign in","sign up","login","log in","register",
    "read more","continue reading","learn more","more","index","archives","categories",
    "next","previous","older posts","newer posts","contact","about","privacy policy",
    "inicio","menú","buscar","suscríbete","suscribirse","iniciar sesión","regístrate",
    "leer más","seguir leyendo","más","índice","archivo","categorías","siguiente","anterior",
    "entradas antiguas","entradas nuevas","contacto","acerca de","política de privacidad",
    "início","menu","pesquisar","pesquisa","inscrever-se","entrar","cadastre-se",
    "leia mais","continuar lendo","mais","índice","arquivo","categorias","seguinte","anterior",
    "postagens antigas","postagens más recientes","contato","sobre","política de privacidade",
    "startseite","menü","suche","abonnieren","anmelden","registrieren","mehr lesen","weiterlesen",
    "mehr","index","archiv","kategorien","weiter","zurück","ältere beiträge","neuere beiträge","kontakt","über uns","datenschutz",
    "accueil","menu","recherche","s'abonner","se connecter","s'inscrire","lire la suite","continuer la lecture",
    "plus","index","archives","catégories","suivant","précédent","articles plus anciens","articles plus récents","contact","à propos","politique de confidentialité",
    "cerca","abbonati","accedi","registrati","leggi di più","continua a leggere","altro",
    "indice","archivio","categorie","successivo","precedente","articoli più vecchi","articoli più recenti","contatti","informazioni","privacy",
    "ana sayfa","menü","ara","abone ol","giriş yap","kayıt ol","daha fazlası","devamını oku","daha fazla",
    "dizin","arşiv","kategoriler","sonraki","önceki","iletişim","hakkında","gizlilik politikası",
    "beranda","menu","cari","berlangganan","masuk","daftar","baca selengkapnya","lanjut membaca","selengkapnya",
    "trang chủ","menu","tìm kiếm","đăng ký","đăng nhập","đăng xuất","đọc thêm","tiếp tục đọc","xem thêm",
    "mục lục","lưu trữ","chuyên mục","tiếp theo","trước","liên hệ","giới thiệu","chính sách bảo mật",
    "главная","меню","поиск","подписаться","войти","регистрация","читать далее","продолжить чтение","ещё",
    "индекс","архив","категории","далее","назад","контакты","о нас","политика конфиденциальности",
    "الرئيسية","القائمة","بحث","اشترك","تسجيل الدخول","إنشاء حساب","اقرأ المزيد","متابعة القراءة","المزيد",
    "فهرس","الأرشيف","التصنيفات","التالي","السابق","اتصل","من نحن","سياسة الخصوصية",
]))
SEPARATORS = (" | ", " - ", " — ", " – ", " » ", " :: ", " : ")
DANGLING_DELIM_RE = re.compile(r"[|–—\-:]\s*$")
DOMAIN_SCAR_RE = re.compile(r"\b(on|at|from|via)\s*(\.|,|:|;|$)")

def scan_nav_noise(text: str,
                   nav_phrases: Iterable[str] = DEFAULT_NAV_PHRASES,
                   sep_bonus: bool = True,
                   tail_bonus: bool = True):
    s = nfkc_casefold(text or "")
    n = max(1, len(s))
    hits = 0; unique = set(); score = 0.0; by_phrase = defaultdict(int)

    def local_bonus(pos):
        bonus = 1.0
        if sep_bonus:
            window = s[max(0,pos-5):min(n,pos+15)]
            if any(sep.strip() in window for sep in SEPARATORS): bonus *= 1.3
        if tail_bonus and pos >= int(0.7*n): bonus *= 1.5
        return bonus

    for p in (nfkc_casefold(p) for p in nav_phrases):
        start = 0
        while True:
            idx = s.find(p, start)
            if idx == -1: break
            hits += 1; unique.add(p); by_phrase[p] += 1
            score += 1.0 * local_bonus(idx)
            start = idx + len(p)

    density = hits / n
    score += 50.0 * density
    return {"hits_total": hits, "hits_unique": len(unique), "by_phrase": dict(by_phrase),
            "density": density, "score": float(score), "norm_len": n}

def repetition_diversity(tokens):
    T = len(tokens)
    if T == 0: return {"ttr":0.0,"max_token_share":1.0,"rep3_share":0.0}
    counts = Counter(tokens)
    max_share = max(counts.values())/T
    grams = [" ".join(tokens[i:i+3]) for i in range(max(0,T-2))]
    rep3 = 0.0 if not grams else 1.0 - (len(set(grams))/len(grams))
    return {"ttr":len(set(tokens))/T, "max_token_share":max_share, "rep3_share":rep3}

def length_score(n, low, high):
    if n<=0: return 0.0
    if low <= n <= high: return 1.0
    dist = min(abs(n-low), abs(n-high))
    return max(0.0, 1.0/(1.0+dist/30.0))

def structural_quality(text: str, field: str, jaccard_similarity_val: Optional[float] = None):
    s = nfkc_casefold(text or "")
    n = len(s)
    toks = tokenize_basic(s)
    div = repetition_diversity(toks)
    s_len = length_score(n, *( (8,160) if field=="title" else (30,300) ))
    s_div = max(0.0, min(1.0, 0.7*div["ttr"] + 0.3*(1.0 - div["max_token_share"])))
    s_rep = 1.0 - div["rep3_share"]
    pen_trunc = 0.2 if DANGLING_DELIM_RE.search(s or "") else 0.0
    pen_scar  = 0.2 if DOMAIN_SCAR_RE.search(s or "") else 0.0
    pen_dup   = 0.0
    if jaccard_similarity_val is not None and field == "desc":
        pen_dup = 0.2 if jaccard_similarity_val >= 0.85 else (0.1 if jaccard_similarity_val >= 0.5 else 0.0)
    base = 0.4*s_len + 0.3*s_div + 0.3*s_rep
    penalty = pen_trunc + pen_scar + pen_dup
    q = max(0.0, min(1.0, base*(1.0 - penalty)))
    return {"q": q, "charlen": n, "ttr": div["ttr"], "max_token_share": div["max_token_share"],
            "rep3_share": div["rep3_share"], "pen_trunc": pen_trunc, "pen_scar": pen_scar, "pen_dup": pen_dup}

def select_meta_with_nav_and_quality(df: pd.DataFrame,
                                     nav_phrases: Iterable[str],
                                     nav_thr: float,
                                     prefer_title: bool,
                                     short_title_word_threshold: int = 8) -> pd.DataFrame:
    df = df.copy()
    titles = df.get("title_meta_domain_removed", pd.Series([""]*len(df))).fillna("")
    descs  = df.get("description_meta_domain_removed", pd.Series([""]*len(df))).fillna("")
    jaccs  = df.get("jaccard_similarity", pd.Series([None]*len(df)))

    rows = []
    for t, d, j in zip(titles, descs, jaccs):
        sc_t = scan_nav_noise(t, nav_phrases)
        sc_d = scan_nav_noise(d, nav_phrases)
        is_noisy_t = (sc_t["score"] >= nav_thr) if (t and t.strip()) else False
        is_noisy_d = (sc_d["score"] >= nav_thr) if (d and d.strip()) else False
        q_t = structural_quality(t, "title", jaccard_similarity_val=j)
        q_d = structural_quality(d, "desc",  jaccard_similarity_val=j)

        both = bool(t.strip()) and bool(d.strip())
        if both:
            is_title_short = len(tokenize_basic(t)) < short_title_word_threshold
            if (not is_noisy_t) and (not is_noisy_d):
                if is_title_short:
                    if q_d["q"] >= 0.5 and (len(tokenize_basic(t)) < len(tokenize_basic(d))):
                        src, txt = "desc", d
                    else:
                        src, txt = "title", t
                elif prefer_title and (q_t["q"] + 0.05 >= q_d["q"]):
                    src, txt = "title", t
                else:
                    src, txt = ("title", t) if q_t["q"] >= q_d["q"] else ("desc", d)
            elif (not is_noisy_t) and is_noisy_d:
                src, txt = "title", t
            elif is_noisy_t and (not is_noisy_d):
                src, txt = "desc", d
            else:
                if abs(sc_t["score"] - sc_d["score"]) > 0.5:
                    src, txt = ("title", t) if sc_t["score"] < sc_d["score"] else ("desc", d)
                else:
                    src, txt = ("title", t) if q_t["q"] >= q_d["q"] else ("desc", d)
        elif t.strip():
            src, txt = "title", t
        elif d.strip():
            src, txt = "desc", d
        else:
            src, txt = "none", ""

        rows.append({
            "title_nav_hits": sc_t["hits_total"], "desc_nav_hits": sc_d["hits_total"],
            "title_nav_score": sc_t["score"], "desc_nav_score": sc_d["score"],
            "is_noisy_title": is_noisy_t, "is_noisy_desc": is_noisy_d,
            "title_q": q_t["q"], "desc_q": q_d["q"],
            "title_charlen": q_t["charlen"], "desc_charlen": q_d["charlen"],
            "title_pen_trunc": q_t["pen_trunc"], "desc_pen_trunc": q_d["pen_trunc"],
            "title_pen_scar": q_t["pen_scar"], "desc_pen_scar": q_d["pen_scar"],
            "title_pen_dup": q_t["pen_dup"], "desc_pen_dup": q_d["pen_dup"],
            "selected_src": src, "selected_text": txt
        })

    extra = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, extra], axis=1)

def _first_non_empty(*cols):
    for c in cols:
        if isinstance(c, str) and c.strip():
            return c
    return ""

def clean_metas(df: pd.DataFrame,
                nav_thr: float = 2.0,
                prefer_title: bool = True,
                short_title_word_threshold: int = 8) -> pd.DataFrame:
    pivot_df = df.copy()

    if "title_meta" not in pivot_df.columns:
        pivot_df["title_meta"] = [
            _first_non_empty(t, ot, tt)
            for t, ot, tt in zip(pivot_df.get("title", ""), pivot_df.get("og:title", ""), pivot_df.get("twitter:title", ""))
        ]
    if "description_meta" not in pivot_df.columns:
        pivot_df["description_meta"] = [
            _first_non_empty(d, od, td, md)
            for d, od, td, md in zip(pivot_df.get("description", ""), pivot_df.get("og:description", ""), pivot_df.get("twitter:description", ""), pivot_df.get("meta_description",""))
        ]

    for col in ["target_domain","title_meta","description_meta"]:
        if col not in pivot_df.columns:
            pivot_df[col] = ""

    pivot_df["title_has_domain_name"] = pivot_df.apply(
        lambda r: check_domain_in_text(r["target_domain"], r["title_meta"]), axis=1)
    pivot_df["desc_has_domain_name"] = pivot_df.apply(
        lambda r: check_domain_in_text(r["target_domain"], r["description_meta"]), axis=1)

    pivot_df["title_meta_cleaned"] = pivot_df["title_meta"].apply(lambda x: dedupe_inside(clean_unit(x)))
    pivot_df["description_meta_cleaned"] = pivot_df["description_meta"].apply(lambda x: dedupe_inside(clean_unit(x)))

    pivot_df["title_meta_domain_removed"] = pivot_df.apply(
        lambda r: remove_domain_from_text(r["target_domain"], r["title_meta_cleaned"], r["title_has_domain_name"]), axis=1)
    pivot_df["description_meta_domain_removed"] = pivot_df.apply(
        lambda r: remove_domain_from_text(r["target_domain"], r["description_meta_cleaned"], r["desc_has_domain_name"]), axis=1)

    pivot_df["is_title_exact_match"] = [
        1 if (str(tgt) == str(ttl)) else 0
        for tgt, ttl in zip(pivot_df["target_domain"].fillna(""), pivot_df["title_meta"].fillna(""))
    ]

    for col in ["title_meta_domain_removed","description_meta_domain_removed"]:
        s = pivot_df[col].fillna("")
        s = s.str.replace("- ", "", regex=False).str.replace(" -", "", regex=False)
        s = s.str.replace("| ", "", regex=False).str.replace(" |", "", regex=False)
        pivot_df[col] = s

    pivot_df["title_meta_domain_removed"] = pivot_df["title_meta_domain_removed"].apply(clean_text_for_distil)
    pivot_df["description_meta_domain_removed"] = pivot_df["description_meta_domain_removed"].apply(clean_text_for_distil)

    pivot_df["jaccard_similarity"] = [
        jaccard_similarity(a, b)
    for a, b in zip(pivot_df["title_meta_domain_removed"], pivot_df["description_meta_domain_removed"])
    ]
    pivot_df["jaccard_similarity"] = np.where(
        (pivot_df["title_meta_domain_removed"] == "") | (pivot_df["description_meta_domain_removed"] == ""),
        np.nan,
        pivot_df["jaccard_similarity"]
    )

    out = select_meta_with_nav_and_quality(
        pivot_df, nav_phrases = DEFAULT_NAV_PHRASES,
        nav_thr = nav_thr, prefer_title = prefer_title,
        short_title_word_threshold = short_title_word_threshold
    )

    out["n_words"] = out["selected_text"].fillna("").apply(lambda x: len(x.split()))
    out["text_len"] = out["selected_text"].fillna("").apply(len)

    return out
