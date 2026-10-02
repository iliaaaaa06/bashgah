"""All Persian UI strings in one place — edit here to change wording."""

APP_TITLE = "دستیار اداری هوشمند"
NAV_CHAT = "گفتگو"
NAV_ADMIN = "مدیریت اسناد"

# ---------- sidebar ----------
STATUS_TITLE = "وضعیت سرویس"
STATUS_READY = "آماده"
STATUS_LLM_DOWN = "مدل زبانی در دسترس نیست"
STATUS_BACKEND_DOWN = "سرور در دسترس نیست"
STATUS_CHUNKS = "{n} قطعه در پایگاه دانش"
STATUS_REFRESH = "بررسی مجدد"

# ---------- chat ----------
CHAT_TITLE = "گفتگو با دستیار"
CHAT_INTRO = (
    "پرسش خود را درباره‌ی آیین‌نامه‌ها، بخشنامه‌ها و امور اداری بنویسید. "
    "پاسخ‌ها فقط بر اساس اسناد ثبت‌شده ارائه می‌شوند."
)
CHAT_SUGGESTIONS = [
    "مرخصی استحقاقی سالانه چند روز است؟",
    "روند ثبت درخواست مأموریت چیست؟",
    "نرخ امروز دلار چقدر است؟",
]
CHAT_PLACEHOLDER = "پیام خود را بنویسید…"
CHAT_THINKING = "در حال جستجو و تهیه‌ی پاسخ…"
CHAT_SETTINGS = "تنظیمات گفتگو"
CHAT_TEMPERATURE = "دما (Temperature)"
CHAT_TEMPERATURE_HELP = "دمای کمتر = پاسخ دقیق‌تر و ثابت‌تر · دمای بیشتر = پاسخ متنوع‌تر. این تنظیم فقط روی گفتگوی شما اثر دارد."
CHAT_CLEAR = "پاک کردن گفتگو"
CHAT_DISCLAIMER = "پاسخ‌ها را پیش از استفاده‌ی رسمی بررسی کنید."

SOURCE_BADGES = {
    "documents": ":green-badge[پاسخ از اسناد]",
    "web": ":blue-badge[پاسخ از جستجوی وب]",
    "none": ":orange-badge[پاسخی یافت نشد]",
}
TEMPERATURE_CAPTION = "دما {t}"
SOURCES_TITLE = "منابع ({n})"
SOURCE_PAGE = "صفحه {p}"
SOURCE_SCORE = "شباهت {s}٪"
SOURCE_UNTITLED = "بدون عنوان"

# ---------- admin ----------
ADMIN_TITLE = "مدیریت اسناد"
ADMIN_LOGIN_TITLE = "ورود مدیر"
ADMIN_LOGIN_HELP = "برای بارگذاری و حذف اسناد، کلید مدیر (ADMIN_API_KEY) را وارد کنید."
ADMIN_KEY_LABEL = "کلید مدیر"
ADMIN_LOGIN = "ورود"
ADMIN_LOGOUT = "خروج"
ADMIN_KEY_EMPTY = "کلید مدیر را وارد کنید."

UPLOAD_TITLE = "بارگذاری اسناد"
UPLOAD_LABEL = "فایل‌های PDF، DOCX یا TXT را انتخاب کنید"
UPLOAD_BUTTON = "بارگذاری و پردازش"
UPLOAD_PROGRESS = "در حال پردازش {n} فایل… (ممکن است برای فایل‌های بزرگ چند دقیقه طول بکشد)"
UPLOAD_CHUNKS = " ({n} قطعه)"

DOCS_TITLE = "اسناد ثبت‌شده"
DOCS_COUNT = "{n} سند"
DOCS_EMPTY = "هنوز سندی بارگذاری نشده است."
DOCS_REFRESH = "بروزرسانی"
DOCS_COL_NAME = "نام فایل"
DOCS_COL_CHUNKS = "تعداد قطعه"
DOCS_COL_DATE = "تاریخ بارگذاری"
DOCS_DELETE = "حذف"
DOCS_DELETE_CONFIRM = "سند «{name}» از پایگاه دانش حذف شود؟"
DOCS_DELETE_YES = "بله، حذف شود"
DOCS_DELETE_NO = "انصراف"

# ---------- errors (network-level; API errors come in Persian from the backend) ----------
ERR_CONNECTION = "ارتباط با سرور برقرار نشد. اجرای سرویس بک‌اند و آدرس BACKEND_URL را بررسی کنید."
ERR_TIMEOUT = "زمان پاسخ‌گویی سرور به پایان رسید. لطفاً دوباره تلاش کنید."
ERR_UNKNOWN = "خطای ناشناخته از سرور دریافت شد."
