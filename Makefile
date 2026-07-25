PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help install install-dev test aula1 aula2 aula3 aula4 relatorio clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make install      -> instala dependências principais"
	@echo "  make install-dev  -> instala dependências principais + pytest"
	@echo "  make test         -> executa a suíte de testes"
	@echo "  make aula1        -> executa a Aula 1"
	@echo "  make aula2        -> executa a Aula 2"
	@echo "  make aula3        -> executa a Aula 3"
	@echo "  make aula4        -> executa a Aula 4"
	@echo "  make relatorio    -> gera relatório final consolidado"
	@echo "  make clean        -> remove caches Python locais"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]

test:
	$(PYTHON) -m pytest

aula1:
	$(PYTHON) aulas/aula_01_coleta_unitarizacao.py

aula2:
	$(PYTHON) aulas/aula_02_sanitizacao_chauvenet.py

aula3:
	$(PYTHON) aulas/aula_03_regressao_ols.py

aula4:
	$(PYTHON) aulas/aula_04_diagnosticos_nbr.py

relatorio:
	$(PYTHON) relatorios/gerador_apostila.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
