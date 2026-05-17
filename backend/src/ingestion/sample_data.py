"""Sample data source — mimic Zalo Legal 2021 schema.

Cho phép test pipeline end-to-end **không cần internet**. Tất cả nội dung
là mô tả ngắn dưới dạng template (KHÔNG phải trích chính xác văn bản pháp luật);
mục đích duy nhất: cấu trúc dữ liệu hợp lệ để smoke test downstream.
"""

from __future__ import annotations

from collections.abc import Iterator

from .base import Article, GoldenQA, IngestionSource

# (law_id, law_title, category, [(article_id, article_title, body)])
_SAMPLE_LAWS: list[tuple[str, str, str, list[tuple[str, str, str]]]] = [
    (
        "45/2019/QH14",
        "Bộ luật Lao động",
        "labor",
        [
            (
                "1",
                "Phạm vi điều chỉnh",
                "Bộ luật Lao động quy định tiêu chuẩn lao động, quyền và nghĩa vụ của các bên.",
            ),
            (
                "13",
                "Hợp đồng lao động",
                "Hợp đồng lao động là sự thỏa thuận giữa người lao động và người sử dụng lao động "
                "về việc làm có trả công, tiền lương, điều kiện lao động.",
            ),
            (
                "41",
                "Bồi thường khi đơn phương chấm dứt hợp đồng trái pháp luật",
                "Trường hợp người sử dụng lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật, "
                "phải nhận người lao động trở lại làm việc và bồi thường ít nhất hai tháng tiền lương.",
            ),
        ],
    ),
    (
        "100/2015/QH13",
        "Bộ luật Hình sự",
        "criminal",
        [
            (
                "8",
                "Khái niệm tội phạm",
                "Tội phạm là hành vi nguy hiểm cho xã hội được quy định trong Bộ luật Hình sự, "
                "do người có năng lực trách nhiệm hình sự hoặc pháp nhân thương mại thực hiện.",
            ),
            (
                "173",
                "Tội trộm cắp tài sản",
                "Người nào trộm cắp tài sản của người khác trị giá từ 2.000.000 đồng trở lên, "
                "thì bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm.",
            ),
        ],
    ),
    (
        "59/2020/QH14",
        "Luật Doanh nghiệp",
        "business",
        [
            (
                "4",
                "Giải thích từ ngữ",
                "Doanh nghiệp là tổ chức có tên riêng, có tài sản, có trụ sở giao dịch, "
                "được thành lập hoặc đăng ký theo quy định của pháp luật nhằm mục đích kinh doanh.",
            ),
            (
                "17",
                "Quyền của doanh nghiệp",
                "Doanh nghiệp có quyền tự do kinh doanh trong những ngành nghề mà luật không cấm.",
            ),
        ],
    ),
    (
        "61/2020/QH14",
        "Luật Đầu tư",
        "business",
        [
            (
                "5",
                "Bảo đảm đầu tư",
                "Nhà nước bảo đảm quyền sở hữu tài sản hợp pháp của nhà đầu tư.",
            ),
            (
                "9",
                "Ngành nghề cấm đầu tư kinh doanh",
                "Cấm các hoạt động đầu tư kinh doanh trong các lĩnh vực ảnh hưởng đến quốc phòng, "
                "an ninh quốc gia, đạo đức xã hội.",
            ),
        ],
    ),
]

# Q&A pairs với golden relevant_articles
_SAMPLE_QA: list[tuple[str, str, list[dict]]] = [
    (
        "q_001",
        "Người lao động bị sa thải trái pháp luật được bồi thường gì?",
        [{"law_id": "45/2019/QH14", "article_id": "41"}],
    ),
    (
        "q_002",
        "Trộm cắp tài sản trị giá 5 triệu bị phạt như thế nào?",
        [{"law_id": "100/2015/QH13", "article_id": "173"}],
    ),
    (
        "q_003",
        "Doanh nghiệp có quyền tự do kinh doanh không?",
        [{"law_id": "59/2020/QH14", "article_id": "17"}],
    ),
    (
        "q_004",
        "Hợp đồng lao động là gì?",
        [{"law_id": "45/2019/QH14", "article_id": "13"}],
    ),
    (
        "q_005",
        "Ngành nghề nào bị cấm đầu tư?",
        [{"law_id": "61/2020/QH14", "article_id": "9"}],
    ),
]


class SampleDataSource(IngestionSource):
    """Mock data source — ~10 articles + 5 Q&A. Always works, không cần internet."""

    @property
    def name(self) -> str:
        return "sample"

    def fetch_corpus(self) -> Iterator[Article]:
        for law_id, law_title, category, articles in _SAMPLE_LAWS:
            for article_id, art_title, art_body in articles:
                yield Article(
                    law_id=law_id,
                    law_title=law_title,
                    article_id=article_id,
                    text=f"Điều {article_id}. {art_title}\n{art_body}",
                    extra={"category": category, "article_title": art_title},
                )

    def fetch_qa(self) -> Iterator[GoldenQA]:
        for qid, question, relevant in _SAMPLE_QA:
            yield GoldenQA(
                question_id=qid,
                question=question,
                relevant_articles=relevant,
            )
