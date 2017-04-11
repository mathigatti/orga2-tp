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
	;global matrix_prod

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

	pop rbp
  ret

 mat_plus_vec:
	push rbp
	mov rbp, rsp

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
	add cl, 3
	and cl, dl						;rcx = w_size mod 4
	cmp cl, 0
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
		loop .not_multiple_of_4
		and dl, 0xFC

	;Inicializo el contador
	.multiple_of_4:
	mov rcx, rdx 					;rcx = w_size
	shr rcx, 2						;Proceso de a 4 elementos

	unpcklps xmm0, xmm0
	unpcklps xmm0, xmm0

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
		loop .ciclo
	
  	pop rbp
  ret

; inputs: rdi, rsi, rdx, rcx, r8
; float* matrix1, float* matrix2, uint n, uint m, float* output

  hadamardProduct:
	push rbp
	mov rbp, rsp

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
	
	pop rbp
  ret

  matrix_prod:
	push rbp
	mov rbp, rsp
	push r12
	push r13
	push r14
	push r15

	;Salvo las dimensiones 
	; mov r10, rdx ; r10 = n
	; mov r11, rcx ; r11 = m
	; mov r12, r8  ; r12 = l
	mov r13, rdx ; En r13 guardo la cte n (en rdx) pues voy a necesitar rdx para multiplicar
	xor r10, r10
	;La cantidad de iteraciones internas es m = rcx
	.i:
		xor r12, r12

		; Calculo desplazamiento en matrix1
		mov rax, r10
		mul ecx
		shl rdx, 32
		;mov edx, eax ; rdx = i * m
		lea r14, [rdx + rax] ; r14 = i * m
		
		.j:
			xor r11, r11
			pxor xmm3, xmm3													;xmm3 = acumulador para el coef r10, r12

			; Calculo desplazamiento en matrix2
			mov r15, r12
			.k:
				movd xmm1, [rdi + 8 * r14]		;xmm1 = matrix1[r10][r11]
				movd xmm2, [rsi + 8 * r15]		;xmm2 = matrix2[r11][r12]
				mulsd xmm1, xmm2
				addsd xmm3, xmm1
				
				inc r14 	;Voy del primer al ultimo elemento de la fila
				add r15, r8
				inc r11
				cmp r11, rcx
				jne .k
			sub r14, rcx

			; Calculo desplazamiento en output
			mov rax, r10
			mul r8d
			shl rdx, 32
			;mov edx, eax ; rdx = i * m
			lea r15, [rdx + rax]
			add r15, r12
			
			movd [r9 + 8 * r15], xmm3
			inc r12
			cmp r12, r8
			jne .j
		inc r10
		cmp r10, r13
		jne .i

	pop r15
	pop r14
	pop r13
	pop r12
	pop rbp
	ret