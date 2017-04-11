; RECORDATORIOS
; inputs: rdi, rsi, rdx, rcx, r8, r9
; preservar: r12, r13, r14, r15, rbx, 
; la pila: rbp, rsp
; devolver cosas por rax o xmmo 
; inputs floats: xmm0, xmm1, ..., xmm7

	global cost_derivative
	global mat_plus_vec
	global update_weight
	global hadamardProduct
	global matrix_prod

; YA IMPLEMENTADAS EN C
	extern fprintf
	extern malloc
	extern free
	extern fclose
	extern fopen

; /** DEFINES **/ 
	%define NULL 		0
	%define TRUE 		1
	%define FALSE 		0
	
	%define LF					10


section .rodata
	UNROLL_AMT  equ   2   ; # of times to unroll the loop


section .data
	msg1: DB '%s', LF, 0	;imprimo string con salto de linea y fin de linea
	msg2: DB 'a',0			;modo append para fopen
	msg3: DB '<oracionVacia>',0
	

section .text


;/** FUNCIONES MATRICIALES **/
;-----------------------------------------------------------

; void cost_derivative(
;	double* res_vec  (rdi)
;	double* target_vec (rsi)
;	double* output	(rdx)
; )
;NOTA: cost_derivative es bastante mas eficiente con SSE2 que con AVR
	cost_derivative:
	push rbp
	mov rbp, rsp

	; ;Calculo la cantidad de elementos total
	; xor rax, rax
	; mov eax, edx
	; mul ecx					;eax = low(n*m) ;edx = high(n*m)
	; shl rdx, 32
	; add rax, rdx			;rax = #pixeles

	;Itero sobre todos los elementos y realizo la operación de SUBPD
	%rep 4
		movupd xmm1, [rdi]	;xmm1 = | x0 | x1 | x2 | x3 |
		movupd xmm2, [rsi]	;xmm2 = | y0 | y1 | y2 | y3 |

		subpd xmm1, xmm2

		movupd [rdx], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add rdx, 16
	%endrep

	movupd xmm1, [rdi]	;xmm1 = | x0 | x1 |
	movupd xmm2, [rsi]	;xmm2 = | y0 | y1 |

	subpd xmm1, xmm2

	movupd [rdx], xmm1

	pop rbp
  ret

  hadamardProduct:
	push rbp
	mov rbp, rsp

	;Calculo la cantidad de pixeles total
	xor rax, rax
	mov eax, edx
	mul ecx					;eax = low(n*m) ;edx = high(n*m)
	shl rdx, 32
	add rax, rdx			;rax = #elementos

	;Chequeo si la cantidad de elementos es par
	mov rdx, 0x1
	and rdx, rax
	jz .A

	.B:
	;Caso impar: opero sobre el primer elemento por separado
	movq xmm1, [rdi]
	movq xmm2, [rsi]

	mulpd xmm1, xmm2

	movq [r8], xmm1
	add rdi, 8
	add rsi, 8
	add r8, 8	
	dec rdx
	jnz .B

	;Inicializo el contador
	.A:
	and al, 0xFE
	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'|

		mulpd xmm1, xmm2

		movupd [r8], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add r8, 16
		sub rax, 2
		jnz .ciclo
	
	pop rbp
  ret

 mat_plus_vec:
	push rbp
	mov rbp, rsp

	;Chequeo si la cantidad de elementos es par
	mov rax, rdx
	mov rdx, 0x1
	and rdx, rax
	jz .A

	.B:
	;Caso impar: opero sobre el primer elemento por separado
	movq xmm1, [rdi]
	movq xmm2, [rsi]
	addsd xmm1, xmm2
	movq [rcx], xmm1
	add rdi, 8
	add rsi, 8
	add rcx, 8	
	dec rdx
	jnz .B

	;Inicializo el contador
	.A:
	and al, 0xFE
	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'|

		addpd xmm1, xmm2

		movupd [rcx], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add rcx, 16
		sub rax, 2
		jnz .ciclo
	
	pop rbp
  ret

;void update_weight(
; 		double* w, 		(rdi) 
;			double* nw, 	(rsi)
;			uint w_size, 	(rdx)	
;			double c    	(xmm0)
;)
  update_weight:
	push rbp
	mov rbp, rsp

	;Calculo w_size mod 4
	xor rcx, rcx
	inc cl
	and cl, dl						;rcx = w_size mod 4
	cmp cl, 0
	jz .multiple_of_4

	;Caso no-multiplo
	.not_multiple_of_4:
		movq xmm1, [rdi]		;xmm1 = w_0
		movq xmm2, [rsi]		;xmm2 = nw_0
		mulsd xmm2, xmm0		;xmm2 = c * nw_0
		subsd xmm1, xmm2		;xmm1 = w_0 - c * nw_0
		movq [rdi], xmm1
		add rdi, 8
		add rsi, 8
		;loop .not_multiple_of_4
		;dec rcx 						;rcx = w_size - 1

	;Inicializo el contador
	.multiple_of_4:
	mov rcx, rdx 						;rcx = w_size
	shr rcx, 1						;Proceso de a 4 elementos
	unpcklpd xmm0, xmm0

	;Itero sobre todos los pesos y realizo la actualizacion
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | w_i | w_i+1 |
		movupd xmm2, [rsi]	;xmm2 = | nw_i| nw_i+1|

		mulpd xmm2, xmm0
		subpd xmm1, xmm2
		movupd [rdi], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		loop .ciclo
	
	pop rbp
  ret

;void matrix_prod(
;	double* matrix1, (rdi) 
;	double* matrix2, (rsi)
; uint n, 				 (rdx)				
;	uint m, 				 (rcx)
; uint l, 				 (r8)	
; double* output   (r9)
;)
;Version backward

matrix_prod:
	;TODO: pasar a xmm (y posteriormente a ymm). Para eso la comprobacion que tengo que hacer es que m sea divisible por 2 (luego por 4).
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
	lea r14, [rdx + rax - 1] ; r14 = i * m

	;Precomputo el offset del ultimo elemento de la anteultima fila de matrix2
	lea rax, [rcx - 1]
	mul r8d
	shl rdx, 32
	lea r13, [rdx + rax - 1]

	;Calculo m mod 2
	mov rbx, 1			
	and rbx, rcx						;rbx = m mod 2
	jnz .i
	dec r14

	.i:
		mov r12, r8
		.j:
			mov r11, rcx		;Uso r11 como contador unicamente
			
			pxor xmm3, xmm3													;xmm3 = acumulador para el coef r10, r12

			; Calculo desplazamiento en matrix2
			lea r15, [r13 + r12]

			;Calculo m mod 2
			mov rbx, 1			
			and rbx, rcx						;rbx = m mod 2			
			; Hago rbx operaciones por separado

			.not_multiple_of_2:
			jz .k
			movsd xmm1, [rdi + 8 * r14]		;xmm1 = matrix1[r10][r11]
			movsd xmm2, [rsi + 8 * r15]		;xmm2 = matrix2[r11][r12]
			mulsd xmm1, xmm2
			addsd xmm3, xmm1
			sub r15, r8 ;Voy del ultimo al primer elemento de la columna
			dec r14 		;Voy del ultimo al primer elemento de la fila
			dec r11
			jz .ready
			dec rbx
			jmp .not_multiple_of_2

			.k:
				movdqu xmm1, [rdi + 8 * r14]		;xmm1 = matrix1[r10][r11]
				movsd xmm2, [rsi + 8 * r15]		;xmm2 = matrix2[r11][r12]
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna
				movsd xmm4, [rsi + 8 * r15]		;xmm2 = matrix2[r11][r12]
				unpcklpd xmm4, xmm2
				mulpd xmm1, xmm4
				addpd xmm3, xmm1
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna
				sub r14, 2 		;Voy del ultimo al primer elemento de la fila
				sub r11, 2
				jnz .k

			.ready:
			add r14, rcx	;Hago esto para situarme de vuelta
										;al final de la fila r10-1
			
			movdqu xmm1, xmm3
			unpckhpd xmm3, xmm3
			addsd xmm3, xmm1
			; Calculo desplazamiento en output
			lea rax, [r10 - 1]
			mul r8d
			shl rdx, 32
			;mov edx, eax ; rdx = i * m
			lea r15, [rdx + rax]
			lea r15, [r15 + r12 - 1]
			
			movq [r9 + 8 * r15], xmm3
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