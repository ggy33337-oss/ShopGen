from poster.models.analysis import AnalyzerResult
from poster.models.poster_schema import PosterSchema


def build_poster_schema(
    user_text,
    analyzer_result: AnalyzerResult,
    poster_type="商业海报",
    campaign="",
    target_audience="",
):
    return PosterSchema(
        product=analyzer_result.product or user_text[:120],
        poster_type=poster_type or "商业海报",
        campaign=campaign or user_text,
        style=analyzer_result.style,
        layout=analyzer_result.layout,
        color_scheme=analyzer_result.color_scheme,
        selling_points=analyzer_result.selling_points,
        target_audience=target_audience or "电商消费者",
    )
