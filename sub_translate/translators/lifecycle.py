"""Единый потокобезопасный жизненный цикл локальных переводчиков."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sub_translate.translators.base import Translator

if TYPE_CHECKING:
    from sub_translate.models import TranslationOptions


@dataclass(slots=True)
class _ActiveLocalTranslator:
    profile_id: str
    unload: Callable[[], None]


class LocalTranslatorLifecycleCoordinator:
    """Сериализует локальный инференс и гарантирует выгрузку при смене профиля."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: _ActiveLocalTranslator | None = None

    @property
    def active_profile_id(self) -> str | None:
        """Возвращает профиль, которому сейчас принадлежит локальный слот."""
        with self._lock:
            return self._active.profile_id if self._active is not None else None

    def create(
        self,
        profile_id: str,
        factory: Callable[[], Translator],
    ) -> CoordinatedLocalTranslator:
        """Создаёт адаптер после подтверждённой выгрузки другого профиля."""
        with self._lock:
            self._switch_away_from_active_locked(profile_id)
            delegate = factory()
            self._active = _ActiveLocalTranslator(profile_id, delegate.unload)
            return CoordinatedLocalTranslator(self, profile_id, delegate)

    def translate_batch(
        self,
        profile_id: str,
        delegate: Translator,
        texts: list[str],
        options: TranslationOptions,
    ) -> list[str]:
        """Удерживает локальный слот на всё время перевода."""
        with self._lock:
            self._switch_away_from_active_locked(profile_id)
            self._active = _ActiveLocalTranslator(profile_id, delegate.unload)
            return delegate.translate_batch(texts, options)

    def unload_profile(self, profile_id: str) -> None:
        """Выгружает профиль, только если он по-прежнему владеет слотом."""
        with self._lock:
            if self._active is None or self._active.profile_id != profile_id:
                return
            self._unload_active_locked()

    def unload_all(self) -> None:
        """Выгружает единственный активный локальный движок."""
        with self._lock:
            self._unload_active_locked()

    def _switch_away_from_active_locked(self, next_profile_id: str) -> None:
        if self._active is None or self._active.profile_id == next_profile_id:
            return
        self._unload_active_locked()

    def _unload_active_locked(self) -> None:
        active = self._active
        if active is None:
            return
        active.unload()
        self._active = None


class CoordinatedLocalTranslator:
    """Адаптер общего контракта, работающий через локальный координатор."""

    __slots__ = ("_coordinator", "_delegate", "_profile_id", "name")

    def __init__(
        self,
        coordinator: LocalTranslatorLifecycleCoordinator,
        profile_id: str,
        delegate: Translator,
    ) -> None:
        self._coordinator = coordinator
        self._profile_id = profile_id
        self._delegate = delegate
        self.name = delegate.name

    def translate_batch(self, texts: list[str], options: TranslationOptions) -> list[str]:
        return self._coordinator.translate_batch(
            self._profile_id,
            self._delegate,
            texts,
            options,
        )

    def unload(self) -> None:
        self._coordinator.unload_profile(self._profile_id)

    def __getattr__(self, name: str) -> Any:
        """Сохраняет доступ к несистемным свойствам конкретного адаптера."""
        return getattr(self._delegate, name)


__all__ = [
    "CoordinatedLocalTranslator",
    "LocalTranslatorLifecycleCoordinator",
]
