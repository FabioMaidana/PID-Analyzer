# Importação das bibliotecas necessárias
from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
# Configura o Matplotlib para o backend 'Agg' (não interativo), essencial para rodar em servidor
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import control
from control import TransferFunction
import os
import socket
import webbrowser
from threading import Timer
from scipy.optimize import minimize
import sympy
import sys

# Desativa completamente a resolução de nome do host
os.environ['SERVER_NAME'] = 'localhost'
socket.getfqdn = lambda *args: 'localhost' 

# Informa a localização das pastas ao backend
# Bloco para lidar com paths quando executado como .exe (PyInstaller)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Se estiver rodando como executável, os templates/static estão em _MEIPASS
    base_path = sys._MEIPASS
    template_dir = os.path.join(base_path, 'templates')
    static_dir = os.path.join(base_path, 'static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
else:
    # Se estiver rodando como script .py normal
    app = Flask(__name__)

# Define 's' do python-control globalmente para uso nas rotas
s = control.TransferFunction.s

@app.route('/')
def inicio():
    return render_template('inicio.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/modelagem', methods=['GET', 'POST'])
def modelagem():
    if request.method == 'POST':
        # Captura valores dos form em string
        raw_dados = request.form.get('dados', '')
        raw_intervalo = request.form.get('intervalo', '')
        raw_funcao = request.form.get('funcao_transf', '')

        # Processa forms
        dados = raw_dados.strip() if raw_dados.strip() else ''
        intervalo = raw_intervalo.strip() if raw_intervalo.strip() else ''
        funcao = raw_funcao.strip() if raw_funcao.strip() else ''

        # --- CASO 1: Input por Dados Pontuais e Intervalo ---
        if dados and intervalo and not funcao:
            try:
                # Cálculo e otimização
                # Converte os dados de entrada (string) em um array numpy
                dados_list = np.array([float(x.replace(',', '.')) for x in dados.split()])
                intervalo_val = float(intervalo.replace(',', '.'))
                
                if intervalo_val <= 0:
                    raise ValueError("Intervalo deve ser maior que zero")

                # Cria o vetor de tempo 'x' com base no intervalo fornecido
                x = np.arange(len(dados_list)) * intervalo_val
                
                # Estimativas iniciais (guesses) para os otimizadores
                K_guess = dados_list[-1] if len(dados_list) > 0 else 1.0
                tau_guess = x[np.argmax(dados_list > K_guess*0.63)]/3 if any(dados_list > K_guess*0.63) else 1.0
                tau_guess = max(0.001, tau_guess) # Garante que tau_guess não seja zero

                print("------------------------------------------------------------------")

                # Tenta o ajuste de 1ª Ordem
                res_1a = minimize(
                    erro_parametros_1a,
                    x0=[K_guess, tau_guess],
                    args=(x, dados_list),
                    method='Nelder-Mead',
                    options={'maxiter': 500}
                )
                erro_1a = res_1a.fun
                print(f"Resultado 1ª Ordem: Erro={erro_1a:.4e}, Params={res_1a.x}")

                # Tenta o ajuste de 2ª Ordem
                res_2a = minimize(
                    erro_parametros_2a,
                    x0=[K_guess, tau_guess, 0.7], # K, tau, zeta
                    args=(x, dados_list),
                    method='Nelder-Mead',
                    options={'maxiter': 500}
                )
                erro_2a = res_2a.fun
                print(f"Resultado 2ª Ordem: Erro={erro_2a:.4e}, Params={res_2a.x}")

                # Compara e decide o melhor modelo
                if erro_1a <= (erro_2a * 1.05):
                    print("Decisão: Usando modelo de 1ª Ordem.")
                    K_otimo, tau_otimo = res_1a.x
                    G = K_otimo / (tau_otimo * s + 1)
                    G_str_parsable = f"({K_otimo})/({tau_otimo}*s + 1)"
                
                else:
                    print("Decisão: Usando modelo de 2ª Ordem.")
                    K_otimo, tau_otimo, zeta_otimo = res_2a.x
                    G = K_otimo / (tau_otimo**2 * s**2 + 2 * zeta_otimo * tau_otimo * s + 1)
                    c_s2 = tau_otimo**2
                    c_s1 = 2 * zeta_otimo * tau_otimo
                    G_str_parsable = f"({K_otimo})/({c_s2}*s**2 + {c_s1}*s + 1)"

                # Simula a resposta ao degrau do modelo otimizado
                t, y_step = control.step_response(G, T=np.max(x))
                # Interpola a resposta simulada para os mesmos pontos de tempo dos dados originais
                y_step_interp = np.interp(x, t, y_step)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x, dados_list, 'bo-', label='Dados Recebidos (Pontos)')
                ax.plot(x, y_step_interp, 'r--', label='Modelo Estimado (Linha)')
                ax.set_xlabel('Tempo (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True)
                img_str = plot_to_base64(fig)
                
                # Gerar string da função G(s) para display
                G_str_display = create_tf_strings(G)

                return render_template('planta_estimada.html',
                                    grafico=img_str,
                                    funcao_display=G_str_display,
                                    funcao_parsable=G_str_parsable,
                                    dados=dados)

            except ValueError as e:
                error_msg = f"<strong>Valor Inválido:</strong> {str(e)}.<br>Verifique se os dados pontuais são números e se o intervalo é maior que zero."
                return render_template('modelagem.html',
                                    error=error_msg), 400
            except Exception as e:
                error_msg = f"<strong>Erro ao processar dados:</strong> {str(e)}.<br>Verifique se os dados de entrada estão formatados corretamente (números separados por espaço)."
                return render_template('modelagem.html',
                                    error=error_msg), 400

        # --- CASO 2: Input por Função de Transferência ---
        elif funcao and not dados and not intervalo:
            try:
                # Converte a string da função de transferência em um objeto 'control'
                G = parse_transfer_function(funcao)

                # Plotagem do gráfico de resposta ao degrau
                # Define um vetor de tempo padrão para a simulação
                t = np.linspace(0, 15, 1000)
                # Calcula a resposta ao degrau da planta informada
                _, y = control.step_response(G, T=t)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(t, y, 'b-', linewidth=2)
                ax.set_xlabel('Tempo (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True)
                img_str = plot_to_base64(fig)

                # Gerar string da função G(s) para display
                G_str_display = create_tf_strings(G)
                
                return render_template('planta_conhecida.html',
                                    grafico=img_str,
                                    funcao_display=G_str_display,
                                    funcao_parsable=funcao)

            except Exception as e:
                error_msg = f"""
                <strong>Erro ao processar a G(s) Teórica:</strong> {str(e)}<br>
                Verifique a sintaxe. Exemplos válidos:
                <ul style='list-style-type: disc; margin-left: 20px;'>
                    <li>1/(s^2+2*s+1)</li>
                    <li>(s+1)/(s^2+3*s+2)</li>
                    <li>1/(s^2+0,5*s+1) &larr; (vírgula ou ponto são aceitos)</li>
                </ul>
                """
                return render_template('modelagem.html', error=error_msg), 400

        # --- CASO 3: Inputs Inválidos ---
        else:
            # Lógica para tratar entradas inválidas ou misturadas
            error_msg = ""
            if dados and not intervalo:
                error_msg = "<strong>Método 1 (Experimental) Incompleto:</strong><br>Para usar 'Dados da Resposta', o 'Intervalo de Amostragem' é obrigatório."
            elif funcao and (dados or intervalo):
                error_msg = "<strong>Entrada Ambígua:</strong><br>Escolha <strong>apenas um</strong> método. Preencha 'Função G(s)' <strong>OU</strong> 'Dados + Intervalo', não ambos."
            else:
                error_msg = "<strong>Nenhum dado fornecido.</strong><br>Preencha os campos para o Método 1 (Experimental) ou Método 2 (Teórico)."

            return render_template('modelagem.html', error=error_msg), 400

    # Método GET (apenas carrega a página de inserção)
    return render_template('modelagem.html')

@app.route('/projeto_controlador')
def projeto_controlador():
    try:
        # Obter dados da URL
        referencia_str = request.args.get('referencia', '1.0').replace(',', '.')
        referencia = float(referencia_str)
        
        G_str_parsable = request.args.get('G_parsable') 
        G_str_display = request.args.get('G_display')
        
        # Reconstruir G(s)
        if G_str_parsable:
            G = parse_transfer_function(G_str_parsable)
        else:
            # Define uma planta padrão caso G_str_parsable seja nulo
            G = control.TransferFunction([1], [1, 1]) 

        # 1. Obter ganhos iniciais e diagnósticos
        Kp, Ki, Kd, diagnostico_sintonia, calculo_sintonia = obter_pid_inicial(G)
        
        # 2. CÁLCULO DO FILTRO (Implementação Final)
        # Calculamos aqui uma única vez para garantir consistência entre Gráfico e Simulação
        tau_f = estimar_tau_f_auto(Kp, Ki, Kd)
        
        fig_proj, ax_proj = plt.subplots(figsize=(10, 6))
        legend_handles = []
        
        # Desenha os eixos (preto, zorder 1)
        ax_proj.axhline(0, color='black', linewidth=1.6, zorder=1)
        ax_proj.axvline(0, color='black', linewidth=1.6, zorder=1)
        
        # Calcula o LGR da Planta G(s) (sem plotar)
        rlist, klist = control.root_locus(G, plot=False)

        # Desenha manualmente os elementos da Planta G(s) (azul, zorder 2 e 3)
        if rlist is not None and rlist.shape[0] > 0:
            lgr_plot = ax_proj.plot(rlist.real[0, :], rlist.imag[0, :], 'b-', 
                                   linewidth=2.0, zorder=2,
                                   label='Caminho LGR (Planta G(s))')
            
            for i in range(1, rlist.shape[0]):
                ax_proj.plot(rlist.real[i, :], rlist.imag[i, :], 'b-', 
                                   linewidth=2.0, zorder=2)
            
            legend_handles.append(lgr_plot[0])
        elif rlist is not None:
            lgr_handle = plt.Line2D([], [], color='blue', linestyle='-', linewidth=2.0,
                                    label='Caminho LGR (Planta G(s))')
            legend_handles.append(lgr_handle)

        poles_G = G.poles()
        if poles_G is not None and len(poles_G) > 0:
            pole_g_handle = ax_proj.plot(poles_G.real, poles_G.imag, 'bx', 
                                         markersize=12, markeredgewidth=3.0, 
                                         label='Polos da Planta (G(s))', zorder=3)
            legend_handles.append(pole_g_handle[0])
        
        zeros_G = G.zeros()
        if zeros_G is not None and len(zeros_G) > 0:
            zero_g_handle = ax_proj.plot(zeros_G.real, zeros_G.imag, 'bo', 
                                         markersize=10, markeredgewidth=2.5, 
                                         markerfacecolor='none', 
                                         label='Zeros da Planta (G(s))', zorder=3)
            legend_handles.append(zero_g_handle[0])

        # 3. LÓGICA DO H(s) COM FILTRO
        zeros_H = []
        polo_H_coords = []
        h_str_display = ""
        
        # Verifica se existe ação integral (Polo na origem)
        if Ki > 1e-9: 
            polo_H_coords.append(0+0j) 
            
        # Verifica se aplicamos a lógica do filtro (Derivativo presente + Filtro válido)
        if Kd > 1e-9 and tau_f > 1e-9:
            # Adiciona o Polo do Filtro (X vermelho extra)
            polo_filtro = -1.0 / tau_f
            polo_H_coords.append(polo_filtro + 0j)

            # Recalcula Zeros considerando a equação completa com filtro:
            # H(s) = ( (Kd + Kp*tau)*s^2 + (Kp + Ki*tau)*s + Ki ) / ( s(1+tau*s) )
            a = Kd + Kp * tau_f
            b = Kp + Ki * tau_f
            c = Ki
            
            zeros_H = np.roots([a, b, c])
            
            # String formatada mostrando o filtro
            h_str_display = f"{Kp:.3f} + {Ki:.3f}/s + {Kd:.3f}s/(1 + {tau_f:.3f}s)".replace('.',',')

        else:
            # Lógica Clássica (Sem Filtro significativo ou apenas PI/P)
            if Kd > 1e-3 and Ki > 1e-3: # PID Ideal
                zeros_H = np.roots([Kd, Kp, Ki])
                h_str_display = f"{Kp:.3f} + {Ki:.3f}/s + {Kd:.3f}s".replace('.',',')
            elif Kd > 1e-3: # PD Ideal
                zeros_H = np.roots([Kd, Kp])
                h_str_display = f"{Kp:.3f} + {Kd:.3f}s".replace('.',',')
            elif Ki > 1e-3: # PI
                zeros_H = np.roots([Kp, Ki])
                h_str_display = f"{Kp:.3f} + {Ki:.3f}/s".replace('.',',')
            else: # P Puro
                h_str_display = f"{Kp:.3f}".replace('.',',')

        # Plotagem dos Polos do Controlador
        if polo_H_coords:
            polo_H_handle = ax_proj.plot(np.real(polo_H_coords), np.imag(polo_H_coords), 
                                         'rx', markersize=12, markeredgewidth=3.0,
                                          label='Polos do Controlador (H(s))', zorder=3) 
            legend_handles.append(polo_H_handle[0])

        # Plotagem dos Zeros do Controlador
        if len(zeros_H) > 0:
            zero_H_handle = ax_proj.plot(zeros_H.real, zeros_H.imag, 'ro', 
                                         markersize=10, markeredgewidth=2.5, 
                                         markerfacecolor='none', 
                                         label='Zeros do Controlador (H(s))', zorder=3)
            legend_handles.append(zero_H_handle[0])
        
        # Calcula e aplica os limites do gráfico (zoom)
        lim_x, lim_y = calculate_plot_limits([poles_G, zeros_G, zeros_H, np.array(polo_H_coords)])
        ax_proj.set_xlim(lim_x)
        ax_proj.set_ylim(lim_y)
        
        # Finaliza o gráfico
        ax_proj.set_xlabel('Eixo Real')
        ax_proj.set_ylabel('Eixo Imaginário')
        ax_proj.legend(handles=legend_handles, loc='best')
        ax_proj.grid(True, which='both', zorder=0)
        
        grafico_projeto_b64 = plot_to_base64(fig_proj)
        
        # Renderiza o template
        return render_template('projeto_controlador.html',
                            diagnostico_sintonia=diagnostico_sintonia,
                            calculo_sintonia=calculo_sintonia,
                            funcao_transferencia_G_parsable=G_str_parsable,
                            funcao_transferencia_G_display=G_str_display,
                            funcao_transferencia_H = h_str_display,
                            grafico_projeto=grafico_projeto_b64,
                            Kp=Kp,
                            Ki=Ki,
                            Kd=Kd,
                            tau_f=tau_f, 
                            referencia=referencia)

    except Exception as e:
        print(f"ERRO GERAL na rota /projeto_controlador: {e}")
        return render_template('erro.html', mensagem=str(e)), 400

@app.route('/analise_controlador', methods=['GET', 'POST'])
def analise_controlador():
    try:
        # Variáveis padrão
        G_str_parsable = None
        G_str_display = None
        tau_f_user = None
        sigma = 0.0
        tau_f_mode = 'auto'
        
        # --- Lógica GET (Carrega a página vindo do Projeto) ---
        if request.method == 'GET':
            # 1. Recebe os dados da URL
            referencia = float(request.args.get('referencia'))
            G_str_parsable = request.args.get('G_parsable') 
            G_str_display = request.args.get('G_display')
            
            Kp = float(request.args.get('Kp'))
            Ki = float(request.args.get('Ki'))
            Kd = float(request.args.get('Kd'))

            # 2. LÓGICA DO FILTRO
            tau_f_param = request.args.get('tau_f')
            
            if tau_f_param and tau_f_param != 'None':
                try:
                    tau_f_user = float(tau_f_param)
                except ValueError:
                    tau_f_user = estimar_tau_f_auto(Kp, Ki, Kd)
            else:
                # Fallback: Só calcula se não veio nada na URL
                tau_f_user = estimar_tau_f_auto(Kp, Ki, Kd)

            # Ruído (Padrão 0.00)
            sigma = float(request.args.get('sigma', 0.00))
            
            if G_str_parsable:
                G = parse_transfer_function(G_str_parsable)
            else:
                G = control.TransferFunction([1], [1, 1]) 

            modo = 0
            centraliza = 1
            diagnostico_sintonia = "<strong>Etapa 4: Sintonia Fina</strong><br>Use os controles abaixo para refinar o projeto inicial."

        # --- Lógica POST (Atualizações via AJAX/Dashboard) ---
        elif request.method == 'POST':
            dados = request.get_json()
            G_str_parsable = dados.get('G_parsable') 
            referencia = float(dados.get('referencia'))
            modo = int(dados.get('modo'))

            # Lê configurações de filtro e ruído
            tau_f_mode = dados.get('tau_f_mode', 'auto')
            tau_f_val_raw = dados.get('tau_f_val', None)
            sigma = float(dados.get('sigma', 0.0))

            # Lógica de Tau Manual vs Auto no POST
            if tau_f_mode == 'manual' and tau_f_val_raw is not None:
                try:
                    tau_f_user = float(str(tau_f_val_raw).replace(',', '.'))
                except ValueError:
                    tau_f_user = 0.0
            else:
                # Se for auto, deixamos None para ser calculado dinamicamente no build_H
                tau_f_user = None 
            
            if modo == 0:  # Ajuste Manual de Ganhos
                Kp = float(dados.get('Kp'))
                Ki = float(dados.get('Ki'))
                Kd = float(dados.get('Kd'))
                
            elif modo == 1:  # Ajuste Assistido (Sintonia Fina)
                fonte_ajuste = dados.get('fonte_ajuste')
                
                Kp_atual = float(dados.get('Kp_atual'))
                Ki_atual = float(dados.get('Ki_atual'))
                Kd_atual = float(dados.get('Kd_atual'))
                
                OS_desejado = float(dados.get('OS'))
                Tr_desejado = float(dados.get('Tr'))

                if G_str_parsable:
                    G = parse_transfer_function(G_str_parsable)
                else:
                    G = control.TransferFunction([1], [1, 1])
                
                # --- Lógica de Otimização Restaurada (Idêntica ao seu original) ---
                metodo_otimizador = None
                
                # Lógica para ajuste de Sobressinal (OS -> Kd)
                if fonte_ajuste == 'OS':
                    Kp_fixo, Ki_fixo, Kd_fixo = Kp_atual, Ki_atual, None
                    chute_inicial, meta_tipo, meta_valor = [Kd_atual], 'OS', OS_desejado
                    args_erro = (G, Kp_fixo, Ki_fixo, Kd_fixo, referencia, meta_tipo, meta_valor)
                    metodo_otimizador = 'Nelder-Mead'

                # Lógica para ajuste de Tempo de Subida (Tr -> Kp)
                elif fonte_ajuste == 'Tr':
                    Kp_fixo, Ki_fixo, Kd_fixo = None, Ki_atual, Kd_atual
                    chute_inicial, meta_tipo = [Kp_atual], 'Tr'
                    
                    # 1. Sistema atual para base de BW
                    H_atual_filtrado = build_H_filtrado(Kp_atual, Ki_atual, Kd_atual)
                    T_atual = control.minreal((G * H_atual_filtrado) / (1 + G * H_atual_filtrado))
                    
                    t_metrics_atual = np.linspace(0, 100, 10000)
                    t_m_atual, y_m_atual = control.step_response(T_atual, T=t_metrics_atual)
                    metricas_atuais = calcular_metricas_desempenho(t_m_atual, y_m_atual * referencia, referencia)
                    tr_atual = metricas_atuais['rise_time']
                    
                    # 2. Largura de banda atual
                    try:
                        bw_atual = control.bandwidth(T_atual)
                        if not np.isfinite(bw_atual):
                            raise Exception("BW não finita")
                    except Exception as e:
                        print(f"Aviso: Falha ao calcular BW atual, usando fallback. {e}")
                        bw_atual = 3.0 / tr_atual if tr_atual > 0.1 else 1.0 
                    
                    # 3. BW alvo
                    if Tr_desejado < 0.01: Tr_desejado = 0.01
                    if tr_atual < 0.01: tr_atual = Tr_desejado
                    
                    bw_alvo = bw_atual * (tr_atual / Tr_desejado)
                    meta_valor = bw_alvo
                    
                    args_erro = (G, Kp_fixo, Ki_fixo, Kd_fixo, referencia, meta_tipo, meta_valor)
                    metodo_otimizador = 'L-BFGS-B' 
                
                else:
                    Kp, Ki, Kd = Kp_atual, Ki_atual, Kd_atual 

                if metodo_otimizador:
                    if metodo_otimizador == 'L-BFGS-B':
                        limites = [(-1000.0, 1000.0)]
                        resultado = minimize(
                            erro_sintonia_unica,
                            x0=chute_inicial,
                            args=args_erro,
                            method=metodo_otimizador,
                            bounds=limites, 
                            options={'maxiter': 50, 'ftol': 1e-3}
                        )
                    else:  # Nelder-Mead
                        resultado = minimize(
                            erro_sintonia_unica,
                            x0=chute_inicial,
                            args=args_erro,
                            method=metodo_otimizador, 
                            options={'maxiter': 50, 'xatol': 1e-3}
                        )
                                            
                    ganho_novo = resultado.x[0]

                    if fonte_ajuste == 'OS':
                        Kp, Ki, Kd = Kp_fixo, Ki_fixo, ganho_novo
                    elif fonte_ajuste == 'Tr':
                        Kp, Ki, Kd = ganho_novo, Ki_fixo, Kd_fixo
                
            tempo_vis = float(dados.get('tempo_simulacao'))
            y_vis = float(dados.get('y_max'))
            centraliza = int(dados.get('centraliza'))

        # --- Lógica Comum (Simulação e Resposta) ---
        
        # 1. Garante objeto G
        if request.method == 'POST' and modo == 0:
             if G_str_parsable: G = parse_transfer_function(G_str_parsable)
             else: G = control.TransferFunction([1], [1, 1])

        # 2. Constrói H(s)
        # Se veio do GET, usa o tau_f que veio do Projeto.
        # Se veio do POST (auto), build_H recalcula baseado nos novos ganhos.
        H = build_H_filtrado(Kp, Ki, Kd, tau_f=tau_f_user)
        
        # 3. Malha Fechada T(s)
        G_loop = control.minreal(G * H)
        T = control.minreal(G_loop / (1 + G_loop))
        
        # 4. Simulação Longa para Métricas
        ts_base = 100.0
        if request.method == 'GET':
            try:
                poles = T.poles()
                if poles is not None and len(poles) > 0:
                    dominant_pole_time = np.max(poles.real)
                    if dominant_pole_time < -1e-3:
                        ts_base = (1 / abs(dominant_pole_time)) * 10
            except:
                pass
            ts_base = max(20.0, min(ts_base, 200.0))

        t_metrics = np.linspace(0, ts_base, 10000)
        t_m, y_m = control.step_response(T, T=t_metrics)
        y_m = y_m * referencia
        metricas = calcular_metricas_desempenho(t_m, y_m, referencia)
        
        # 5. Ajuste de Zoom
        if centraliza == 1:
            tempo_vis = metricas['settling_time'] * 2.2
            y_vis = (referencia + metricas['overshoot_val']) * 1.1
            if y_vis < referencia * 1.1:
                y_vis = referencia * 1.1
            if tempo_vis <= 0.1:
                tempo_vis = 10.0
            if y_vis <= 0.1:
                y_vis = referencia * 1.5 
        
        # 6. Simulação Visual Final
        t_vis = np.linspace(0, tempo_vis, 1000)
        t_v, y_v = control.step_response(T, T=t_vis)
        y_v = y_v * referencia

        # Injeção de Ruído
        if request.method == 'GET':
            seed = int(request.args.get('seed', 42))
        else:
            seed = 42

        rng = np.random.default_rng(seed)
        n = rng.normal(0.0, sigma, size=t_v.shape)

        # Caminho do ruído: Tn(s) = - G(s)H(s) / (1 + G(s)H(s))
        Tn = -control.feedback(G * H, 1)
        resp = control.forced_response(Tn, T=t_v, U=n)
        if isinstance(resp, tuple) and len(resp) == 3:
            _, y_n, _ = resp
        else:
            _, y_n = resp

        # Saída total
        y_v = y_v + y_n

        fig_step, ax_step = plt.subplots(figsize=(10, 5))
        label_grafico = f"PID (Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f})".replace('.',',')
        ax_step.plot(t_v, y_v, 'g-', linewidth=2, label=label_grafico)
        ax_step.axhline(y=referencia, color='r', linestyle='--', label='Referência')
        ax_step.set_xlim([0, tempo_vis])
        ax_step.set_ylim([0, y_vis])
        ax_step.set_xlabel('Tempo (s)')
        ax_step.set_ylabel('Saída')
        ax_step.legend(loc='best')
        ax_step.grid(True, alpha=0.3)
        grafico_step_b64 = plot_to_base64(fig_step)

        # Preparação para Retorno (Display correto do Filtro)
        # Precisamos saber qual tau_f foi realmente usado para mostrar na tela/JSON
        if tau_f_user is None:
            tau_disp_val = estimar_tau_f_auto(Kp, Ki, Kd)
        else:
            tau_disp_val = tau_f_user

        if tau_disp_val is not None and tau_disp_val > 1e-4:
            h_str = f"{Kp:.3f} + {Ki:.3f}/s + {Kd:.3f}s/(1 + {tau_disp_val:.3f}s)"
            tau_f_display_str = f"{tau_disp_val:.4f}".replace('.', ',')
        else:
            h_str = f"{Kp:.3f} + {Ki:.3f}/s + {Kd:.3f}s"
            tau_f_display_str = ""
            
        sigma_display_str = f"{sigma:.3f}".replace('.', ',')

        # --- Retorno GET (HTML) ---
        if request.method == 'GET':
            return render_template(
                'analise_controlador.html',
                diagnostico_sintonia=diagnostico_sintonia,
                funcao_transferencia_G_parsable=G_str_parsable,
                funcao_transferencia_G_display=G_str_display,
                funcao_transferencia_H=h_str.replace('.',','),

                grafico_step=grafico_step_b64,
                Kp_str=f"{Kp:.3f}".replace('.',','),
                Ki_str=f"{Ki:.3f}".replace('.',','),
                Kd_str=f"{Kd:.3f}".replace('.',','),
                Kp_float=Kp,
                Ki_float=Ki,
                Kd_float=Kd,
                referencia=f"{referencia}".replace('.',','),
                tempo_simulacao=f"{tempo_vis:.1f}".replace('.',','),
                y_max=f"{y_vis:.1f}".replace('.',','),
                overshoot_percent=f"{metricas['overshoot_percent']:.2f}".replace('.',','),
                rise_time=f"{metricas['rise_time']:.3f}".replace('.',','),
                settling_time=f"{metricas['settling_time']:.3f}".replace('.',','),
                steady_state_error=f"{metricas['steady_state_error']:.3f}".replace('.',','),
                modo=modo,
                OS=f"{metricas['overshoot_percent']:.2f}".replace('.',','), 
                Tr=f"{metricas['rise_time']:.3f}".replace('.',','),
                tau_f_display=tau_f_display_str,
                sigma_display=sigma_display_str,
                tau_f_mode=tau_f_mode
            )
                                
        # --- Retorno POST (JSON) ---
        elif request.method == 'POST':
            return jsonify({
                'status': 'success',
                'grafico_step': grafico_step_b64,
                'metricas': {
                    'overshoot_val': f"{metricas['overshoot_val']:.3f}".replace('.',','),
                    'overshoot_percent': f"{metricas['overshoot_percent']:.2f}".replace('.',','),
                    'rise_time': f"{metricas['rise_time']:.3f}".replace('.',','),
                    'settling_time': f"{metricas['settling_time']:.3f}".replace('.',','),
                    'steady_state_error': f"{metricas['steady_state_error']:.3f}".replace('.',',')
                },
                'ganhos': {
                    'Kp_str': f"{Kp:.3f}".replace('.',','),
                    'Ki_str': f"{Ki:.3f}".replace('.',','),
                    'Kd_str': f"{Kd:.3f}".replace('.',','),
                    'Kp_float': Kp,
                    'Ki_float': Ki,
                    'Kd_float': Kd
                },
                'ganhos_str': {
                    'h_str': h_str.replace('.',',')
                },
                'visualizacao': {
                    'tempo_simulacao': f"{tempo_vis:.1f}".replace('.',','),
                    'y_max': f"{y_vis:.1f}".replace('.',',')
                },
                'filtro_ruido': {
                    'tau_f': tau_f_display_str,
                    'tau_f_mode': tau_f_mode,
                    'sigma': sigma_display_str
                }
            })
    except Exception as e:
        print(f"ERRO GERAL na rota /analise_controlador: {e}") 
        if request.method == 'GET':
            return render_template('erro.html', mensagem=str(e)), 400
        elif request.method == 'POST':
            return jsonify({'status': 'error', 'message': str(e)}), 400

def erro_parametros_1a(parametros, x, y):
    """Função de custo para otimizar um modelo de 1ª ordem (K / (tau*s + 1))."""
    K, tau = parametros
    try:
        # Adiciona restrições para estabilidade do otimizador
        if tau <= 0.01:
            return float('inf')
            
        G = K / (tau * s + 1)
        t, resp = control.step_response(G, T=np.max(x)*1.2)
        y_pred = np.interp(x, t, resp)
        # Retorna o Erro Quadrático Médio (MSE)
        return np.mean((y - y_pred) ** 2)
    except:
        return float('inf')

def erro_parametros_2a(parametros, x, y):
    """Função de custo para otimizar um modelo de 2ª ordem (K / (tau²s² + 2*zeta*tau*s + 1))."""
    K, tau, zeta = parametros
    try:
        G = K / (tau**2 * s**2 + 2 * zeta * tau * s + 1)
        t, resp = control.step_response(G, T=np.max(x)*1.2)
        y_pred = np.interp(x, t, resp)
        # Retorna o Erro Quadrático Médio (MSE)
        return np.mean((y - y_pred) ** 2)
    except:
        return float('inf')

def create_tf_strings(G_tf):
    """Cria strings de exibição (ex: s²) e 'parsable' (ex: s**2) a partir de um objeto TF."""
    # Usa Sympy para manipulação simbólica e formatação bonita
    s_sympy = sympy.symbols('s')
    num_coeffs = G_tf.num[0][0]
    den_coeffs = G_tf.den[0][0]
    # Normaliza o denominador para ter '1' como coeficiente de maior ordem
    leading_coeff = den_coeffs[0]
    
    if not np.isclose(leading_coeff, 0):
        num_coeffs_norm = num_coeffs / leading_coeff
        den_coeffs_norm = den_coeffs / leading_coeff
    else:
        num_coeffs_norm = num_coeffs
        den_coeffs_norm = den_coeffs
        
    num_coeffs_rounded = [round(float(c), 3) for c in num_coeffs_norm]
    den_coeffs_rounded = [round(float(c), 3) for c in den_coeffs_norm]
    
    num_poly = sympy.Poly(num_coeffs_rounded, s_sympy)
    den_poly = sympy.Poly(den_coeffs_rounded, s_sympy)
    
    # Cria a expressão simbólica
    G_sympy = num_poly / den_poly
    
    parsable_str = str(G_sympy)
    parsable_str = parsable_str.replace('1.000*', '').replace('1.0*', '')
    
    # Formata a string para exibição amigável (HTML)
    display_str = parsable_str.replace('**', '^')
    display_str = display_str.replace('*', '')
    display_str = display_str.replace('^2', '²').replace('^3', '³').replace('^4', '⁴')
    
    # Converte o ponto decimal em vírgula para exibição no HTML
    display_str = display_str.replace('.', ',')
    
    return display_str

def obter_pid_inicial(G):
    """
    Função principal (Dispatcher) que analisa G(s) e despacha para o método
    de sintonia correto, RETORNANDO DIAGNÓSTICO E CÁLCULO.
    """
    try:
        # Simplificar a planta
        G = control.minreal(G, verbose=False)

        # Análise de Risco (polos/zeros)
        poles = G.poles()
        zeros = G.zeros()
        
        Kp, Ki, Kd = 1.0, 0.1, 0.01 # Ganhos de fallback
        diagnostico_str = "Não foi possível classificar."
        calculo_str = "Usando ganhos de segurança (1.0, 0.1, 0.01)."

        if poles is None or len(poles) == 0:
            diagnostico_str = "Diagnóstico: <strong>Planta Estável (Ganho Puro)</strong>."
            Kp, Ki, Kd, calculo_str = sintonizar_zn_estavel(G)
        else:
            # Calcular condições de risco
            n_unstable_poles = np.sum(poles.real > 1e-6)
            n_rhp_zeros = 0
            if zeros is not None and len(zeros) > 0:
                n_rhp_zeros = np.sum(zeros.real > 1e-6)
            n_integrators = np.sum(np.abs(poles) < 1e-6) 
            n_oscillators = np.sum((np.abs(poles.real) < 1e-6) & (np.abs(poles.imag) > 1e-6))

            # Triagem (Hierarquia de Prioridade)

            # Prioridade: Planta Instável (Polo RHP)
            if n_unstable_poles > 0:
                if n_rhp_zeros > 0:
                    # Caso: Instável E Fase Não-Mínima (Pior Cenário)
                    diagnostico_str = "Diagnóstico: <strong>Planta Instável e de Fase Não-Mínima (Polo e Zero no RHP).</strong>"
                    calculo_str = "Este é um sistema de controle avançado. Sintonia automática não suportada. Ganhos definidos como 0."
                    Kp, Ki, Kd = 0.0, 0.0, 0.0
                else:
                    # Caso: Apenas Instável (Polos RHP)
                    if n_unstable_poles == 1:
                        diagnostico_str = "Diagnóstico: <strong>Planta Instável (1 Polo no RHP)</strong>."
                        Kp, Ki, Kd, calculo_str = sintonizar_planta_instavel(G)
                    else:
                        diagnostico_str = f"Diagnóstico: <strong>Planta Instável ({n_unstable_poles} Polos no RHP)</strong>."
                        Kp, Ki, Kd, calculo_str = sintonizar_planta_instavel_2polos(G)
            
            # Prioridade: Fase Não-Mínima (Zero RHP) - mas Estável
            elif n_rhp_zeros > 0:
                if (n_integrators > 0 or n_oscillators > 0):
                     # Caso: Marginalmente estável e Fase Não-Mínima
                    diagnostico_str = "Diagnóstico: <strong>Planta Marginalmente Estável e de Fase Não-Mínima (Zero no RHP).</strong>"
                    calculo_str = "Este é um sistema de controle avançado. Sintonia automática não suportada. Ganhos definidos como 0."
                    Kp, Ki, Kd = 0.0, 0.0, 0.0
                else:
                    # Caso Padrão de Fase Não-Mínima (Estável)
                    diagnostico_str = "Diagnóstico: <strong>Estável, Fase Não-Mínima (Zero no RHP)</strong>."
                    Kp, Ki, Kd, calculo_str = sintonizar_fase_nao_minima(G)

            # Prioridade: Casos Especiais (Marginalmente Estáveis)
            elif n_integrators > 0:
                diagnostico_str = f"Diagnóstico: <strong>Planta Integradora (Nível {n_integrators})</strong>."
                Kp, Ki, Kd, calculo_str = sintonizar_integrador(G, n_integrators)
            
            elif n_oscillators > 0:
                diagnostico_str = "Diagnóstico: <strong>Planta Oscilatória (Polos em &pm;j&omega;)</strong>."
                Kp, Ki, Kd, calculo_str = sintonizar_oscilador(G)

            # Prioridade: Planta "Normal" (Estável, Fase Mínima)
            else:
                diagnostico_str = "Diagnóstico: <strong>Planta Estável (Padrão)</strong>."
                Kp, Ki, Kd, calculo_str = sintonizar_zn_estavel(G)
        if not np.isfinite(Kp): Kp = 1.0
        if not np.isfinite(Ki): Ki = 0.0 
        if not np.isfinite(Kd): Kd = 0.0
            
        print(f"Ganhos Iniciais Calculados: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
        return Kp, Ki, Kd, diagnostico_str, calculo_str
        
    except Exception as e:
        print(f"ERRO CRÍTICO em obter_pid_inicial: {e}. Retornando fallback seguro.")
        diagnostico_str = f"<strong>Erro Crítico na Análise:</strong><br>A sintonia automática falhou.<br><strong>Mensagem:</strong> {e}"
        calculo_str = "Usando ganhos de segurança (1,0, 0,1, 0,01)."
        return (1.0, 0.1, 0.01, diagnostico_str, calculo_str)

def sintonizar_fase_nao_minima(G):
    """
    Sintonia PI SUPER CONSERVADORA para plantas estáveis com zeros no RHP.
    O objetivo não é ser rápido, é ser ESTÁVEL e um ponto de partida seguro.
    *** CORREÇÃO: Ganhos negativos são permitidos e necessários para plantas com K_dc negativo. ***
    """
    calculo_str = "<strong>Método: PI Super-Conservador (para Fase Não-Mínima)</strong><br>"
    try:
        # Obtém o ganho estático (DC gain) da planta
        K_dc = control.dcgain(G)
        if np.abs(K_dc) < 1e-6:
             K_dc = 0.01 # Evita divisão por zero, assume ganho pequeno
        
        # Usa um ganho proporcional muito baixo, 10% do inverso do ganho DC
        Kp = 0.1 / K_dc
        # Usa uma ação integral muito lenta (Ti = 10 segundos)
        Ti = 10.0
        Ki = Kp / Ti
        Kd = 0.0

        calculo_str += "Planta: Fase Não-Mínima (Zero RHP).<br>"
        calculo_str += f"Ganho DC (K) estimado: {K_dc:.2f}<br>".replace('.',',')
        calculo_str += "Meta: PI super-conservador (baixo ganho) para garantir estabilidade.<br>"
        calculo_str += f"<strong>Resultados: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}</strong>".replace('.',',')

        return Kp, Ki, Kd, calculo_str
        
    except Exception as e:
        calculo_str += f"Cálculo falhou ({e}). Usando fallback."
        # Retorna ganhos positivos de fallback
        return 0.1, 0.01, 0.0, calculo_str

def sintonizar_integrador(G, n_integrators):
    poles = G.poles()
    # Separa os polos integradores (na origem) dos polos estáveis
    stable_poles = poles[(np.abs(poles) > 1e-6) & (poles.real < -1e-6)]
    calculo_str = "<strong>Método: Alocação de Polos (PD/P)</strong><br>"
    
    try:
        # Caso 1: Integrador puro (K/s)
        if n_integrators == 1 and len(stable_poles) == 0:
            # Calcula o ganho K (multiplicando por 's' para cancelar o polo na origem)
            K_dc_gain = control.dcgain(G * s)
            if not np.isfinite(K_dc_gain) or np.abs(K_dc_gain) < 1e-6: K_dc_gain = 1.0
            ts_desejado = 4.0
            polo_desejado = 4.0 / ts_desejado
            Kp = polo_desejado / K_dc_gain
            Ki, Kd = 0.0, 0.0
            calculo_str += f"Planta: K/s (K={K_dc_gain:.2f})<br>Meta: Polo em s = -{polo_desejado:.2f}<br>Fórmula: Kp = Polo / K<br><strong>Resultados: Kp={Kp:.2f}, Ki=0, Kd=0</strong>".replace('.',',')
        
        # Caso 2: Integrador + polo estável (K / (s(s+a)))
        elif n_integrators == 1 and len(stable_poles) > 0:
            a = np.abs(stable_poles[0].real)
            # Calcula o ganho K isolando os polos
            K = control.dcgain(G * s * (s+a))
            if not np.isfinite(K) or np.abs(K) < 1e-6: K = 1.0
            wn_desejado = 2.0 * a 
            zeta_desejado = 1.0
            Kp = (wn_desejado**2) / K
            Kd = (2 * zeta_desejado * wn_desejado - a) / K
            Ki = 0.0
            calculo_str += f"Planta: K / (s(s+a)) (K={K:.2f}, a={a:.2f})<br>Meta: &zeta;=1,0, &omega;n=2a={wn_desejado:.2f}<br>Fórmulas: Kp=&omega;n&sup2;/K, Kd=(2&zeta;&omega;n-a)/K<br><strong>Resultados: Kp={Kp:.2f}, Ki=0, Kd={Kd:.2f}</strong>".replace('.',',')

        # Caso 3: Integrador duplo ou superior (K/s^2)
        elif n_integrators >= 2:
            K = control.dcgain(G * s**2)
            if not np.isfinite(K) or np.abs(K) < 1e-6: K = 1.0
            ts_desejado = 4.0 
            zeta_desejado = 1.0
            wn_desejado = 4.0 / ts_desejado 
            Kp = (wn_desejado**2) / K
            Kd = (2 * zeta_desejado * wn_desejado) / K
            Ki = 0.0
            calculo_str += f"Planta: K/s&sup2; (K={K:.2f})<br>Meta: &zeta;=1,0, &omega;n={wn_desejado:.2f}<br>Fórmulas: Kp=&omega;n&sup2;/K, Kd=2&zeta;&omega;n/K<br><strong>Resultados: Kp={Kp:.2f}, Ki=0, Kd={Kd:.2f}</strong>".replace('.',',')
        
        return Kp, Ki, Kd, calculo_str
    
    except Exception as e:
        calculo_str += f"Cálculo falhou ({e}). Usando fallback."
        return 1.0, 0.0, 0.1, calculo_str

def sintonizar_oscilador(G):
    calculo_str = "<strong>Método: Alocação de Polos (PD)</strong><br>"
    try:
        poles = G.poles()
        # Encontra o polo oscilatório (no eixo imaginário)
        osc_pole = poles[(np.abs(poles.real) < 1e-6) & (poles.imag > 1e-7)][0]
        omega_n_planta = np.abs(osc_pole.imag)
        # Calcula o ganho K isolando os polos oscilatórios
        K = control.dcgain(G * (s**2 + omega_n_planta**2))
        if not np.isfinite(K) or np.abs(K) < 1e-6: K = 1.0
        
        # Define uma meta de alocação (um pouco mais rápida)
        wn_desejado = 1.2 * omega_n_planta
        zeta_desejado = 0.707
        Kp = (wn_desejado**2 - omega_n_planta**2) / K
        Kd = (2 * zeta_desejado * wn_desejado) / K
        Ki = 0.0
        calculo_str += f"Planta: K/(s&sup2;+&omega;&sup2;) (K={K:.2f}, &omega;={omega_n_planta:.2f})<br>Meta: &zeta;=0,707, &omega;n_novo=1,2&omega;={wn_desejado:.2f}<br>Fórmulas: Kp=(&omega;n_novo&sup2;-&omega;&sup2;)/K, Kd=2&zeta;&omega;n_novo/K<br><strong>Resultados: Kp={Kp:.2f}, Ki=0, Kd={Kd:.2f}</strong>".replace('.',',')
        return Kp, Ki, Kd, calculo_str
    
    except Exception as e:
        calculo_str += f"Cálculo falhou ({e}). Usando fallback."
        return 1.0, 0.0, 1.0, calculo_str

def sintonizar_primeira_ordem(G):
    """
    Calcula um controlador PI "inteligente" para plantas de 1ª ordem
    usando a sintonia IMC (Internal Model Control) para cancelamento de polo.
    """
    calculo_str = "<strong>Método: Sintonia PI (IMC para 1ª Ordem)</strong><br>"
    try:
        # Forma da Planta: K / (tau*s + 1)
        K_dc = control.dcgain(G)
        
        # Polo é -1/tau
        polo = G.poles()[0].real
        tau = -1.0 / polo
        
        # Heurística IMC: tau_i = tau (para cancelar o polo)
        # Kp = tau / (K_dc * lambda), onde lambda é a "agressividade".
        # Vamos escolher lambda = tau / 3 (3x mais rápido que a planta)
        
        # Kp = tau / (K_dc * (tau / 3)) = 3 / K_dc
        Kp = 3.0 / K_dc
        
        # tau_i = tau
        Ki = Kp / tau 
        Kd = 0.0
        
        calculo_str += f"Planta: K/(Ts+1) (K={K_dc:.2f}, T={tau:.2f})<br>".replace('.',',')
        calculo_str += "Meta: Cancelar polo (Ti = T) e 3x mais rápido.<br>"
        calculo_str += "Fórmulas: Kp=3/K, Ki=Kp/T<br>"
        calculo_str += f"<strong>Resultados: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}</strong>".replace('.',',')
        
        return Kp, Ki, Kd, calculo_str

    except Exception as e:
        calculo_str += f"Cálculo falhou ({e}). Usando fallback."
        return 1.0, 0.1, 0.0, calculo_str

def sintonizar_zn_estavel(G):
    # Roteia o método ZN baseado na ordem da planta e no sucesso da sintonia
    poles = G.poles()
    if poles is None:
        try:
            return sintonizar_zn_malha_aberta(G) # Fallback
        except Exception as e_ma:
            print(f"Sintonia ZN (Malha Aberta) falhou: {e_ma}. Usando fallback final.")
            diagnostico_str = "Diagnóstico: Planta estável."
            calculo_str = f"ZN (Curva de Reação) falhou. Usando fallback seguro.<br>Erro: {e_ma}"
            return 1.0, 0.1, 0.01, calculo_str

    n_poles = len(poles)

    if n_poles == 1:
        # 1ª Ordem: Usa IMC (PI) - Mais rápido e estável que ZN
        print("Sintonia ZN: Detectado 1 polo. Usando sintonizar_primeira_ordem (IMC).")
        return sintonizar_primeira_ordem(G)
    
    # ZN-Oscilação (Malha Fechada) Só funciona para 3ª ordem ou superior.
    # Para 2ª ordem, devemos ir DIRETO para ZN-Curva de Reação (Malha Aberta).
    elif n_poles == 2:
        print(f"Sintonia ZN: Detectado {n_poles} polos. Usando ZN (Curva de Reação).")
        try:
            return sintonizar_zn_malha_aberta(G)
        except Exception as e_ma:
            # Se ZN (Malha Aberta) falhar (ex: Sanity Check),
            # usa o fallback de segurança final.
            print(f"Sintonia ZN (Malha Aberta) falhou: {e_ma}. Usando fallback final.")
            diagnostico_str = "Diagnóstico: Planta estável."
            calculo_str = f"ZN (Curva de Reação) falhou. Usando fallback seguro.<br>Erro: {e_ma}"
            return 1.0, 0.1, 0.01, calculo_str
    
    # Se for 3ª Ordem ou superior, tenta ZN (Oscilação) primeiro.
    else: # n_poles >= 3
        print(f"Sintonia ZN: Detectado {n_poles} polos. Tentando ZN (Oscilação).")
        try:
            # Tenta o método de oscilação (malha fechada)
            return sintonizar_zn_oscilacao(G)
        except Exception as e_osc:
            # Se ZN (Oscilação) falhar, usa o ZN (Curva de Reação) como fallback.
            print(f"Sintonia ZN (Oscilação) falhou: {e_osc}. Usando ZN (Curva de Reação) como fallback.")
            try:
                return sintonizar_zn_malha_aberta(G)
            except Exception as e_ma:
                # Se ZN (Malha Aberta) também falhar,
                # usa o fallback de segurança final.
                print(f"Sintonia ZN (Malha Aberta) também falhou: {e_ma}.")
                diagnostico_str = "Diagnóstico: Planta estável."
                calculo_str = f"ZN (Oscilação e Curva de Reação) falharam. Usando fallback seguro.<br>Erro: {e_ma}"
                return 1.0, 0.1, 0.01, calculo_str
                
def sintonizar_zn_malha_aberta(G):
    calculo_str = "<strong>Método: Ziegler-Nichols (Curva de Reação)</strong><br>"
    try:
        t = np.linspace(0, 50, 2000)
        t_resp, y_resp = control.step_response(G, T=t)
        K_dc = y_resp[-1]
        if np.isinf(K_dc) or K_dc == 0: raise ValueError("Planta integradora.")
        dy_dt = np.gradient(y_resp, t_resp)
        idx_max_slope = np.argmax(dy_dt)
        max_slope = dy_dt[idx_max_slope]
        if max_slope < 1e-6: raise ValueError("Slope nulo.")
        t_inflex = t_resp[idx_max_slope]; y_inflex = y_resp[idx_max_slope]
        L = t_inflex - (y_inflex / max_slope)
        T_line = (K_dc - y_inflex) / max_slope + t_inflex; T = T_line - L
        
        # Garante que L e T não sejam zero para evitar divisão por zero
        if L < 1e-3 or T < 1e-3: L = max(1e-3, L); T = max(1e-3, T)

        # Se o atraso (L) for irrealisticamente pequeno (< 1% da const. de tempo T),
        # o método ZN (Curva) não é confiável e irá explodir os ganhos.
        if (L / T) < 0.01:
            raise Exception(f"Atraso (L={L:.4f}) muito pequeno em relação a T ({T:.4f}). Método ZN-Curva de Reação é inadequado.".replace('.',','))

        Kp = 1.2 * (T / L); Ti = 2.0 * L; Td = 0.5 * L
        Ki = Kp / Ti; Kd = Kp * Td
        
        calculo_str += f"Ganho (K) estimado: {K_dc:.3f}<br>".replace('.',',')
        calculo_str += f"Atraso (L) estimado: {L:.3f} s<br>".replace('.',',')
        calculo_str += f"Constante (T) estimada: {T:.3f} s<br>".replace('.',',')
        calculo_str += "Fórmulas: Kp=1,2*T/L, Ti=2,0*L, Td=0,5*L<br>"
        calculo_str += f"<strong>Resultados: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}</strong>".replace('.',',')
        return Kp, Ki, Kd, calculo_str
        
    except Exception as e:
        # Se o cálculo falhar (ex: Slope nulo ou Sanity Check), lança a exceção
        # para que o 'try/except' em 'sintonizar_zn_estavel' possa pegá-la.
        raise Exception(f"Cálculo ZN-Curva de Reação falhou. Erro: {e}")

def sintonizar_planta_instavel(G):
    calculo_str = "<strong>Método: LGR (Espelhamento de 1 Polo)</strong><br>"
    try:
        poles = G.poles()
        poles_instaveis = poles[poles.real > 1e-6]
        if len(poles_instaveis) == 0: return sintonizar_zn_estavel(G)
        
        # Foca no polo instável mais à direita
        polo_instavel_real = np.max(poles_instaveis.real)
        
        # Define o zero do PD para "espelhar" o polo instável
        z_pd = polo_instavel_real 
        
        # Malha para o LGR: G_loop = G(s) * (s + z_pd)
        G_loop = G * (s + z_pd)
        
        # Meta: Mover o polo para um local estável, ex: -polo_instavel_real
        s_desejado = -polo_instavel_real * 1.0
        
        # Lida com o caso em que s_desejado está em cima de um polo/zero
        if np.abs(s_desejado + z_pd) < 1e-6: 
            s_desejado = -polo_instavel_real * 1.1 # Desloca um pouco

        # Avalia a FT no ponto 's_desejado' para encontrar o ganho
        val_em_s_desejado = control.evalfr(G_loop, s_desejado)
        
        if val_em_s_desejado is None or np.abs(val_em_s_desejado) < 1e-9:
            Kd = 5.0 # Fallback
        else:
            # Do LGR: Ganho K = 1 / |G_loop(s_desejado)|
            Kd = 1.0 / np.abs(val_em_s_desejado)
        
        # H(s) = Kp + Kd*s = Kd*(s + Kp/Kd) -> Kp/Kd = z_pd -> Kp = Kd * z_pd
        Kp = Kd * z_pd
        
        # A estratégia de espelhamento de polo simples usa um PD puro.
        Ki = 0.0
        
        calculo_str += f"Polo Instável (Planta): s = +{polo_instavel_real:.2f}<br>".replace('.',',')
        calculo_str += f"Zero do Controlador (PD): s = -{z_pd:.2f}<br>".replace('.',',')
        calculo_str += f"Meta: Mover polo para s = {s_desejado:.2f}<br>".replace('.',',')
        calculo_str += "Fórmulas: Kp=Kd*z_pd, Ki=0 (PD Puro)<br>"
        
        # Checagem de estabilidade (com PD)
        H_final = Kp + Kd*s # Sem Ki
        T_final = control.minreal( (G * H_final) / (1 + G * H_final) )
        
        # Se o controlador PD projetado ainda for instável, usa fallback
        if np.any(T_final.poles().real > 1e-6):
            calculo_str += "AVISO: PD não estabilizou. Usando fallback.<br>"
            return 10.0, 0.0, 5.0, calculo_str # Fallback (ainda PD)

        calculo_str += f"<strong>Resultados: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}</strong>".replace('.',',')
        return Kp, Ki, Kd, calculo_str
        
    except Exception as e:
        calculo_str += f"Cálculo falhou ({e}). Usando fallback."
        return 10.0, 0.0, 5.0, calculo_str # Fallback (PD)

def sintonizar_planta_instavel_2polos(G):
    calculo_str = "<strong>Método: LGR (Espelhamento de 2 Polos)</strong><br>"
    try:
        poles = G.poles()
        unstable_poles = poles[poles.real > 1e-6]
        if len(unstable_poles) < 2: return sintonizar_planta_instavel(G)
        # Extrai os dois polos instáveis
        p_a = unstable_poles[0].real; p_b = unstable_poles[1].real
        # Estratégia: PID onde os zeros cancelam os polos instáveis
        K_base = 10.0
        Kd_final = K_base
        Kp_final = K_base * (p_a + p_b)
        Ki_final = K_base * p_a * p_b
        
        calculo_str += f"Polos Instáveis (Planta): s = +{p_a:.2f}, s = +{p_b:.2f}<br>".replace('.',',')
        calculo_str += f"Zeros do Controlador (PID): s = -{p_a:.2f}, s = -{p_b:.2f}<br>".replace('.',',')
        calculo_str += "Fórmulas (K=10): Kd=K, Kp=K(p_a+p_b), Ki=K(p_a*p_b)<br>".replace('.',',')
        
        # Testa a estabilidade do controlador projetado (com filtro)
        tau_f_auto = estimar_tau_f_auto(Kp_final, Ki_final, Kd_final)
        H_final_filtrado = build_H_filtrado(Kp_final, Ki_final, Kd_final, tau_f=tau_f_auto)
        T_final = control.minreal( (G * H_final_filtrado) / (1 + G * H_final_filtrado) )
        # Se a sintonia com K=10 falhar, tenta um ganho maior
        if np.any(T_final.poles().real > 1e-6):
            K_base = 50.0; Kd_final = K_base; Kp_final = K_base * (p_a + p_b); Ki_final = K_base * p_a * p_b
            calculo_str += "Sintonia inicial instável. Aumentando K para 50.<br>"

        calculo_str += f"<strong>Resultados: Kp={Kp_final:.2f}, Ki={Ki_final:.2f}, Kd={Kd_final:.2f}</strong>".replace('.',',')
        return Kp_final, Ki_final, Kd_final, calculo_str

    except Exception as e:
        calculo_str += f"Cálculo falhou ({e}). Usando fallback."
        return 20.0, 10.0, 10.0, calculo_str

def sintonizar_zn_oscilacao(G):
    """
    Implementa ZN (Malha Fechada / Oscilação). 
    *SÓ DEVE SER CHAMADO PARA PLANTAS DE 3ª ORDEM OU SUPERIOR.*
    """
    calculo_str = "<strong>Método: Ziegler-Nichols (Oscilação)</strong><br>"
    
    # Inicia a busca binária pelo Ganho Último (Ku)
    Ku = 0.1; Ku_low = 0.0; Ku_high = np.inf
    # Loop de iteração para encontrar Ku
    for _ in range(50):
        T = (G * Ku) / (1 + G * Ku)
        # Verifica os polos da malha fechada com o ganho Ku atual
        poles = T.poles()
        if poles is None or len(poles) == 0: Ku *= 0.5; continue
        if np.any(poles.real > 1e-6):
            # Instável: Ku é muito alto. Diminui o teto.
            Ku_high = Ku; Ku = (Ku_low + Ku) / 2
        else:
            # Estável: Ku é muito baixo. Aumenta o piso.
            Ku_low = Ku
            if Ku_high == np.inf: Ku *= 2.0
            else: Ku = (Ku_high + Ku) / 2
        if np.abs(Ku_high - Ku_low) < 1e-3 or Ku > 1e6: break
    Ku = Ku_low
    
    # Se Ku não for encontrado (nunca oscila), falha o método
    if Ku <= 1e-3 or Ku > 1e6:
         raise Exception("ZN (Oscilação) falhou (planta não oscila).")

    T_final = (G * Ku) / (1 + G * Ku)
    poles_final = T_final.poles()
    # Encontra o polo que cruzou o eixo imaginário
    polo_critico = poles_final[np.argmin(np.abs(poles_final.real))]
    freq_oscilacao_rad = np.abs(polo_critico.imag)
    
    if freq_oscilacao_rad < 1e-3:
         raise Exception("ZN (Oscilação) falhou (frequência nula).")
    
    # Calcula o Período Último (Pu) a partir da frequência de oscilação
    Pu = 2 * np.pi / freq_oscilacao_rad

    # Aplica as fórmulas de ZN para Malha Fechada (PID)
    Kp = 0.6 * Ku; Ti = 0.5 * Pu; Td = 0.125 * Pu
    Ki = Kp / Ti; Kd = Kp * Td
    
    calculo_str += f"Ganho Último (Ku) encontrado: {Ku:.3f}<br>".replace('.',',')
    calculo_str += f"Período Último (Pu) encontrado: {Pu:.3f} s<br>".replace('.',',')
    calculo_str += "Fórmulas: Kp=0,6*Ku, Ti=0,5*Pu, Td=0,125*Pu<br>"
    calculo_str += f"<strong>Resultados: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}</strong>".replace('.',',')
    return Kp, Ki, Kd, calculo_str

def erro_sintonia_unica(ganho_variavel_lista, G, Kp_fixo, Ki_fixo, Kd_fixo, referencia, meta_tipo, meta_valor):
    """
    Função de custo para otimizar UM único ganho (Kp ou Kd),
    mantendo os outros dois fixos.
    """
    try:
        ganho_variavel = ganho_variavel_lista[0] # minimize() envia como lista
        
        # Reconstrói os ganhos
        if Kp_fixo is None: Kp = ganho_variavel
        else: Kp = Kp_fixo
        
        if Ki_fixo is None: Ki = ganho_variavel
        else: Ki = Ki_fixo
        
        if Kd_fixo is None: Kd = ganho_variavel
        else: Kd = Kd_fixo

        # Monta o controlador H(s) com filtro
        H = build_H_filtrado(Kp, Ki, Kd)
        # Monta a malha fechada T(s)
        T = control.minreal( (G * H) / (1 + G * H) )
        
        # Checa estabilidade
        poles = T.poles()
        if poles is None or np.any(poles.real > 0.001): 
            return 1e9 # Penalidade por instabilidade
        
        # Lógica para Sobressinal (OS) - USA a simulação de degrau
        if meta_tipo == 'OS':
            t_m = np.linspace(0, 100, 1500) 
            t_m, y_m = control.step_response(T, T=t_m)
            y_m = y_m * referencia
            
            if len(y_m) < 10 or np.max(y_m) < 1e-6:
                return 1e5
                
            metricas = calcular_metricas_desempenho(t_m, y_m, referencia)
            erro_real = metricas['overshoot_percent']
        
        # Lógica para Tempo de Subida (Tr) - USA a Largura de Banda (Bandwidth)
        elif meta_tipo == 'Tr':
            try:
                # Calcula a largura de banda do novo sistema T(s)
                bw_novo = control.bandwidth(T)
                if not np.isfinite(bw_novo):
                    bw_novo = 0.0
            except Exception as e:
                print(f"Aviso: Falha ao calcular BW na otimização: {e}")
                bw_novo = 0.0 # Penaliza se o cálculo da BW falhar
                
            erro_real = bw_novo
            # 'meta_valor' neste caso é o 'bw_alvo' calculado na rota
        
        else:
            return 1e4 # Meta desconhecida
        
        # Adiciona uma pequena penalidade para ganhos grandes
        custo = (erro_real - meta_valor)**2 + (abs(ganho_variavel) * 0.001)
        return custo

    except Exception as e:
        # Retorna um custo alto se a simulação falhar
        print(f"ERRO em erro_sintonia_unica: {e}")
        return 1e7

def build_H_filtrado(Kp, Ki, Kd, tau_f=None):
    """
    Constrói o controlador PID H(s) com filtro derivativo dinâmico.
    Regras:
    - Kd ~ 0  -> retorna P/PI (sem D)
    - tau_f = 0 -> derivativo CLÁSSICO (Kd*s)
    - tau_f > 0 -> derivativo FILTRADO (Kd*s)/(1+tau_f*s)
    - tau_f = None -> calcula automático (heurística)
    """
    # Sem termo D
    if Kd < 1e-3:
        return Kp + Ki/s

    # Se vier None, calcula automático (heurística)
    if tau_f is None:
        try:
            # mesmo cálculo usado antes
            if Ki > 1e-3:
                zeros_H = np.roots([Kd, Kp, Ki])
            else:
                zeros_H = np.roots([Kd, Kp])
            if len(zeros_H) > 0:
                zeros_filtrados = zeros_H[np.abs(zeros_H) > 1e-3]
                if len(zeros_filtrados) > 0:
                    zero_dom = np.min(np.abs(zeros_filtrados))
                else:
                    zero_dom = 0.1
            else:
                zero_dom = 0.1
            polo_filtro = max(10.0, zero_dom * 10.0)
            tau_f = 1.0 / polo_filtro
        except Exception as e:
            print(f"Erro ao calcular tau_f dinâmico: {e}. Usando fallback 0.03.")
            tau_f = 0.03

    # Se tau_f == 0 => PID clássico
    if tau_f <= 0:
        H_d = Kd * s
        return Kp + Ki/s + H_d

    # Caso geral: PID filtrado
    H_d = (Kd * s) / (1 + tau_f * s)
    return Kp + Ki/s + H_d

def estimar_tau_f_auto(Kp, Ki, Kd):
    """Replica a heurística de tau_f usada no filtro, para exibirmos o valor também na string."""
    try:
        if Kd < 1e-3:
            return 0.0
        # zeros do controlador (PD ou PID)
        if Ki > 1e-3:
            zeros_H = np.roots([Kd, Kp, Ki])  # PID: Kd*s^2 + Kp*s + Ki
        else:
            zeros_H = np.roots([Kd, Kp])      # PD: Kd*s + Kp
        if len(zeros_H) > 0:
            zeros_filtrados = zeros_H[np.abs(zeros_H) > 1e-3]
            if len(zeros_filtrados) > 0:
                zero_dom = np.min(np.abs(zeros_filtrados))
            else:
                zero_dom = 0.1
        else:
            zero_dom = 0.1
        polo_filtro = max(10.0, zero_dom * 10.0)
        return 1.0 / polo_filtro
    except Exception as e:
        print(f"Aviso: falha ao estimar tau_f para display: {e}")
        return 0.03  # fallback amigável

def calcular_metricas_desempenho(t, y, referencia):
    
    # Limpa 'inf' e 'nan' da simulação para evitar erros
    y_finite = y[np.isfinite(y)]
    if len(y_finite) == 0:
        # Se a simulação falhou completamente
        return {
            'overshoot_val': np.inf, 'overshoot_percent': np.inf,
            'rise_time': 0, 'settling_time': t[-1],
            'steady_state_error': np.inf,
        }
    
    overshoot_val = max(0, np.max(y_finite) - referencia)
    overshoot_percent = 100 * overshoot_val / referencia if referencia != 0 else 0
    
    target_10 = 0.1 * referencia
    target_90 = 0.9 * referencia
    t10 = 0.0
    t90 = t[-1]
    
    try:
        # Tenta encontrar os tempos t10 e t90
        idx_10 = np.where(y_finite >= target_10)[0][0]
        t10 = t[idx_10]
        idx_90 = np.where(y_finite[idx_10:] >= target_90)[0][0] + idx_10
        t90 = t[idx_90]
        rise_time = t90 - t10
    except IndexError:
        rise_time = t[-1] # Fallback se nunca atingir 90%
    
    # Calcular o valor final de regime estacionário (y_ss) PRIMEIRO.
    window_size = max(5, int(len(y_finite) * 0.1))
    y_final_window = y_finite[-window_size:]
    y_ss = np.mean(y_final_window) # Define nova referência

    # Calcular o Erro Estacionário (baseado em y_ss)
    erro_absoluto = np.abs(y_ss - referencia)
    steady_state_error = 100 * erro_absoluto / referencia if referencia != 0 else 0    
    if not np.isfinite(y_ss):
        steady_state_error = np.inf

    # Calcular o Tempo de Acomodação (2% do valor FINAL y_ss)
    settling_time = t[-1]
    
    # Define a faixa de tolerância como 2% do valor de regime (y_ss)
    # Se y_ss for 0, usa 2% da referência como fallback.
    if np.abs(y_ss) > 1e-6:
        tolerance_band = 0.02 * np.abs(y_ss)
    else:
        tolerance_band = 0.02 * np.abs(referencia) # Fallback

    # Lida com o caso em que a referência é 0 e y_ss é 0
    if tolerance_band < 1e-9:
        tolerance_band = 0.02 # Fallback para uma banda pequena
        
    # Compara com o centro y_ss
    within_tolerance = np.abs(y - y_ss) <= tolerance_band
    
    out_of_tolerance_indices = np.where(~within_tolerance)[0]
    if len(out_of_tolerance_indices) > 0:
        last_out_index = out_of_tolerance_indices[-1]
        if last_out_index < len(t) - 1:
            settling_time = t[last_out_index + 1]
    else:
        # Se todos os pontos estão dentro, o tempo é zero (ou o primeiro ponto)
        settling_time = t[0]

    return {
        'overshoot_val': overshoot_val,
        'overshoot_percent': overshoot_percent,
        'rise_time': rise_time,
        'settling_time': settling_time,
        'steady_state_error': steady_state_error,
    }

def plot_to_base64(fig):
    # Converte um gráfico matplotlib para base64
    # Salva o gráfico em um buffer de bytes na memória
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    # Rebimbina o buffer para o início antes de ler
    buf.seek(0)
    # Codifica os bytes em base64 e decodifica para string utf-8
    return base64.b64encode(buf.read()).decode('utf-8')

def parse_transfer_function(expr_str):
    """
    Converte com segurança uma string de função de transferência em um objeto control.TransferFunction
    usando sympy.
    """
    try:
        s_sympy = sympy.symbols('s')
        
        expr_clean = expr_str.replace(',', '.').replace('^', '**').replace(' ', '')
        
        # Converte a string limpa em uma expressão simbólica Sympy
        expr_sympy = sympy.sympify(expr_clean)
        # Separa o numerador e o denominador da expressão
        num_sympy, den_sympy = expr_sympy.as_numer_denom()
        
        num_poly = sympy.Poly(num_sympy, s_sympy)
        den_poly = sympy.Poly(den_sympy, s_sympy)
        
        # Extrai os coeficientes polinomiais como listas de floats
        num_coeffs = [float(c) for c in num_poly.all_coeffs()]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()]
        
        if not num_coeffs: num_coeffs = [0.0]
        if not den_coeffs: raise ValueError("Denominador não pode ser zero.")

        # Cria o objeto TransferFunction da biblioteca 'control'
        return control.TransferFunction(num_coeffs, den_coeffs)
        
    # Captura erros de parsing do Sympy
    except (sympy.SympifyError, TypeError, ValueError) as e:
        raise ValueError(f"Não foi possível interpretar a expressão '{expr_str}'. Detalhes: {e}")
    except Exception as e:
        raise Exception(f"Erro inesperado no parsing da função: {e}")

def calculate_plot_limits(points_list, default_padding=1.0):
    """
    Calcula limites de plotagem (xlim, ylim) com base em uma lista de arrays de polos/zeros.
    """
    try:
        # Garante que a origem (0,0) esteja sempre incluída no cálculo
        valid_points = [np.array([0+0j])] 
        for p_array in points_list:
            if p_array is not None and len(p_array) > 0:
                valid_points.append(p_array)
        
        # Junta todos os polos e zeros em um único array
        all_points = np.concatenate(valid_points)
        
        real_min, real_max = np.min(all_points.real), np.max(all_points.real)
        imag_min, imag_max = np.min(all_points.imag), np.max(all_points.imag)

        # Define padding (distância mínima da borda)
        real_padding = max(default_padding, (real_max - real_min) * 0.2)
        imag_padding = max(default_padding, (imag_max - imag_min) * 0.2)
        
        # Garante que o padding não seja zero se min==max
        if real_padding < 1e-6: real_padding = default_padding
        if imag_padding < 1e-6: imag_padding = default_padding

        lim_x = [real_min - real_padding, real_max + real_padding]
        lim_y = [imag_min - imag_padding, imag_max + imag_padding]
        
        # Garante que os eixos tenham uma largura/altura mínima (centralizados)
        if lim_y[0] > -default_padding and lim_y[1] < default_padding:
             lim_y = [-default_padding, default_padding]
             
        if lim_x[0] > -default_padding and lim_x[1] < default_padding:
             lim_x = [-default_padding, default_padding]

        return lim_x, lim_y
        
    except Exception as e:
        print(f"Erro ao calcular limites: {e}. Usando fallback.")
        return ([-2.0, 2.0], [-2.0, 2.0]) # Limites de segurança

def open_browser():
    # Evita que o browser abra duas vezes (uma para o reloader do Flask)
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        # Espera 2 segundos para o servidor Flask iniciar antes de abrir o browser
        Timer(2, lambda: webbrowser.open("http://localhost:5000")).start()

if __name__ == '__main__':
    open_browser()
    app.run(debug=True, host='127.0.0.1', port=5000)