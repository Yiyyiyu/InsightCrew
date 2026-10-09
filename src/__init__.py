# InsightCrew — 多Agent协作技术调研与选型平台

# 关键：必须在任何 `import crewai` 之前完成环境引导。
# CrewAI 在 import 期初始化遥测，会向 ~/.config/crewai 写文件；
# 在受限文件沙箱下该写入会无限阻塞（而非抛异常），导致 import 永久挂死。
from src.core import bootstrap as _bootstrap  # noqa: F401
