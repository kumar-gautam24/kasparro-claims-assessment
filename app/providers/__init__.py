from app.config import Settings


def make_provider(settings: Settings):
    if settings.provider == "gemini":
        from app.providers.gemini import GeminiProvider
        return GeminiProvider(settings.model, settings.temperature)
    if settings.provider == "ollama":
        from app.providers.ollama import OllamaProvider
        return OllamaProvider(settings.model, settings.temperature)
    raise ValueError(f"unknown provider {settings.provider!r}")
