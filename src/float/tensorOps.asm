; RECORDATORIOS
; inputs: rdi, rsi, rdx, rcx, r8, r9
; preservar: r12, r13, r14, r15, rbx, 
; la pila: rbp, rsp
; devolver cosas por rax o xmmo 
; inputs floats: xmm0, xmm1, ..., xmm7

	global cost_derivative
	global mat_plus_vec
	global update_weight
	global hadamard_product
	global matrix_prod

; YA IMPLEMENTADAS EN C
	extern fprintf
	extern malloc
	extern free
	extern fclose
	extern fopen

; /** DEFINES **/ 
	%define NULL        0
	%define TRUE        1
	%define FALSE       0
	
	%define LF          10

section .text

;/** FUNCIONES MATRICIALES **/
;-----------------------------------------------------------

; void cost_derivative(
;	double* res_vec  (rdi)
;	double* target_vec (rsi)
;	double* output	(rdx)
; )

  cost_derivative:
	; ;Calculo la cantidad de elementos total
	; xor rax, rax
	; mov eax, edx
	; mul ecx					;eax = low(n*m) ;edx = high(n*m)
	; shl rdx, 32
	; add rax, rdx			;rax = #pixeles

	;Itero sobre todos los elementos y realizo la operación de SUBPD
	%rep 2
		movups xmm1, [rdi]	;xmm1 = | x0 | x1 | x2 | x3 |
		movups xmm2, [rsi]	;xmm2 = | y0 | y1 | y2 | y3 |

		subps xmm1, xmm2

		movups [rdx], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add rdx, 16
	%endrep

	movq xmm1, [rdi]	;xmm1 = | x0 | x1 |
	movq xmm2, [rsi]	;xmm2 = | y0 | y1 |

	subps xmm1, xmm2

	movq [rdx], xmm1
  ret

 mat_plus_vec:
	;Chequeo si la cantidad de elementos es multiplo de 4
	mov rax, rdx
	mov rdx, 0x3
	and rdx, rax
	jz .A

	.B:
	;Caso multiplo de 4: opero sobre el primer elemento por separado
	movd xmm1, [rdi]
	movd xmm2, [rsi]
	addss xmm1, xmm2
	movd [rcx], xmm1
	add rdi, 4
	add rsi, 4
	add rcx, 4	
	dec rdx
	jnz .B

	;Inicializo el contador
	.A:
	and al, 0xFC
	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 | px2 | px3 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'| px2'| px3'|

		addps xmm1, xmm2

		movupd [rcx], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add rcx, 16
		sub rax, 4
		jnz .ciclo
  ret

;void update_weight(
; 		double* w, 		(rdi) 
;			double* nw, 	(rsi)
;			uint w_size, 	(rdx)	
;			double c    	(xmm0)
;)

  update_weight:
	;Calculo w_size mod 4
	xor rcx, 0x3
	and cl, dl						;rcx = w_size mod 4
	jz .multiple_of_4

	;Caso no-multiplo
	.not_multiple_of_4:
		movd xmm1, [rdi]		;xmm1 = w_0
		movd xmm2, [rsi]		;xmm2 = nw_0
		mulss xmm2, xmm0		;xmm2 = c * nw_0
		subss xmm1, xmm2		;xmm1 = w_0 - c * nw_0
		movd [rdi], xmm1
		add rdi, 4
		add rsi, 4
		dec cl
		jnz .not_multiple_of_4

	;Inicializo el contador
	.multiple_of_4:
	shr rdx, 2						;Proceso de a 4 elementos

	vbroadcastss xmm0, xmm0

	;Itero sobre todos los pesos y realizo la actualizacion
	.ciclo:
		movups xmm1, [rdi]	;xmm1 = | w_i | w_i+1 | w_i+2 | w_i+3 |
		movups xmm2, [rsi]	;xmm2 = | nw_i| nw_i+1| nw_i+2| nw_i+3|

		mulps xmm2, xmm0
		subps xmm1, xmm2
		movups [rdi], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		dec rdx
		jnz .ciclo
  ret

; inputs: rdi, rsi, rdx, rcx, r8
; float* matrix1, float* matrix2, uint n, uint m, float* output

  hadamard_product:
	;Calculo la cantidad de pixeles total
	xor rax, rax
	mov eax, edx
	mul ecx					;eax = low(n*m) ;edx = high(n*m)
	shl rdx, 32
	add rax, rdx			;rax = #elementos

	;Chequeo si la cantidad de elementos es multiplo de 4
	mov rdx, 0x3
	and rdx, rax
	jz .A

	.B:
	;Caso multiplo de 4: opero sobre el primer elemento por separado
	movd xmm1, [rdi]
	movd xmm2, [rsi]

	mulss xmm1, xmm2

	movd [r8], xmm1
	add rdi, 4
	add rsi, 4
	add r8, 4	
	dec rdx
	jnz .B

	;Inicializo el contador
	.A:
	and al, 0xFC
	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 | px2 | px3 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'| px2'| px3'|

		mulps xmm1, xmm2

		movupd [r8], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add r8, 16
		sub rax, 4
		jnz .ciclo
  ret

matrix_prod:
	push rbp
	mov rbp, rsp
	push r12
	push r13
	push r14
	push r15
	push rbx
	push rbx ; Cambiar esto por el sub correspondiente

	mov r10, rdx ; r10 = n

	; Calculo desplazamiento del ultimo elemento
	; de matrix1
	mov rax, r10
	mul ecx
	shl rdx, 32
	lea r14, [rdx + rax - 1] ; r14 = n * m - 1

	;Precomputo el offset del ultimo elemento de la anteultima fila de matrix2
	xor rax, rax
	lea eax, [ecx - 1]
	mul r8d
	shl rdx, 32
	lea r13, [rdx + rax - 1] ; r13 = (m-1) * l - 1

	;Calculo m mod 4
	mov rbx, 3
	and rbx, rcx	;rbx = m mod 4
	jnz .i
	sub r14, 3
	.i:
		mov r12, r8
		.j:
			mov r11, rcx		;Uso r11 como contador unicamente
			pxor xmm3, xmm3		;xmm3 = acumulador para el coef r10, r12

			; Calculo desplazamiento en matrix2
			lea r15, [r13 + r12]

			;Calculo m mod 4
			mov rbx, 3
			and rbx, rcx						;rbx = m mod 4
			jz .k
			
			; Hago rbx operaciones por separado
			.not_multiple_of_4:
				movss xmm1, [rdi + 4 * r14]	;xmm1 = matrix1[r10][r11]
				movss xmm2, [rsi + 4 * r15] ;xmm2 = matrix2[r11][r12]
				mulss xmm1, xmm2
				addss xmm3, xmm1
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna
				dec r14 	;Voy del ultimo al primer elemento de la fila
				dec r11
				jz .nm4
				dec rbx
				jz .p
				jmp .not_multiple_of_4
			.p:
				sub r14, 3 ; me posiciono en r10-4
			.k:
				movdqu xmm1, [rdi + 4 * r14]		;xmm1 = matrix1[r10][r11]

				movss xmm6, [rsi + 4 * r15]
				movss xmm2, xmm6		;xmm2 = matrix2[r11][r12]
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna
				pslldq xmm2, 4

				movss xmm6, [rsi + 4 * r15]
				movss xmm2, xmm6		;xmm2 = matrix2[r11][r12]
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna
				pslldq xmm2,4

				movss xmm6, [rsi + 4 * r15]
				movss xmm2, xmm6		;xmm2 = matrix2[r11][r12]
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna
				pslldq xmm2,4

				movss xmm6, [rsi + 4 * r15]
				movss xmm2, xmm6		;xmm2 = matrix2[r11][r12]
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna

				mulps xmm1, xmm2
				addps xmm3, xmm1
				sub r11, 4
				jz .ready
				sub r14, 4 		;Voy del ultimo al primer elemento de la fila
				jmp .k

			.ready:
				dec r14 ; Esto junto con el "add r14, rcx" de mas abajo me posiciona en r10-1

				mov rbx, 3
				and rbx, rcx						;rbx = m mod 4
				jnz .nm4
				sub r14,3 ; Si es multiplo de 4 entonces no se le van a restar 3 para posicionarse en el lugar correcto
							  ; lo hago ahora entonces para posicionarme en r10-4
				.nm4:
				add r14, rcx	;Hago esto para situarme de vuelta
								;al final de la fila r10-1

				movdqu xmm1, xmm3
				psrldq xmm3, 4
				addss xmm1, xmm3
				psrldq xmm3, 4
				addss xmm1, xmm3
				psrldq xmm3, 4
				addss xmm1, xmm3
				; Calculo desplazamiento en output
				lea rax, [r10 - 1]
				mul r8d
				shl rdx, 32
				;mov edx, eax ; rdx = i * m
				lea r15, [rdx + rax]
				lea r15, [r15 + r12 - 1]
				
				movss [r9 + 4 * r15], xmm1
				dec r12
				jnz .j
		sub r14d, ecx
		dec r10
		jnz .i

	pop rbx ; Cambiar esto por el add correspondiente 
	pop rbx
	pop r15
	pop r14
	pop r13
	pop r12
	pop rbp
	ret