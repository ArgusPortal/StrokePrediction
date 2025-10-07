"""
Script para iniciar o servidor FastAPI com verificações de ambiente
"""
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_port_available(port=8000):
    """Verifica se a porta está disponível"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False

def wait_for_server(url="http://127.0.0.1:8000/ping", timeout=30):
    """Aguarda o servidor ficar disponível"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("STROKE PREDICTION API - SERVER STARTER")
    print("=" * 60)
    
    # Verificar se a porta está disponível
    if not check_port_available(8000):
        print("\n❌ ERRO: Porta 8000 já está em uso!")
        print("\nPara resolver:")
        print("1. Feche qualquer servidor rodando na porta 8000")
        print("2. Ou use: netstat -ano | findstr :8000")
        print("3. Depois: taskkill /PID <PID> /F")
        sys.exit(1)
    
    print("\n✓ Porta 8000 disponível")
    print("\nIniciando servidor FastAPI...")
    print("-" * 60)
    
    # Iniciar servidor sem reload para evitar duplicação de métricas
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "run_api_prod.py"],
            cwd=Path(__file__).parent
        )
        
        # Aguardar servidor inicializar (aumentado para 60s)
        print("\nAguardando servidor inicializar...")
        if wait_for_server(timeout=60):
            print("\n" + "=" * 60)
            print("✓ SERVIDOR INICIADO COM SUCESSO!")
            print("=" * 60)
            print("\n📍 URLs disponíveis:")
            print("   • API Docs:     http://127.0.0.1:8000/docs")
            print("   • Health Check: http://127.0.0.1:8000/health")
            print("   • Ping:         http://127.0.0.1:8000/ping")
            print("   • Metrics:      http://127.0.0.1:8000/metrics")
            print("\n⌨  Pressione Ctrl+C para parar o servidor")
            print("=" * 60)
            
            # Manter rodando
            process.wait()
        else:
            print("\n❌ ERRO: Servidor não respondeu no tempo esperado")
            print("Verifique os logs em: logs/api.log")
            process.kill()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹  Parando servidor...")
        if process:
            process.terminate()
            process.wait()
        print("✓ Servidor parado com sucesso")
        print("✓ Servidor parado com sucesso")
    except Exception as e:
        print(f"\n❌ ERRO ao iniciar servidor: {e}")
        sys.exit(1)
