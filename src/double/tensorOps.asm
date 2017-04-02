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

	;Calculo w_size mod 8
	mov rcx, 7
	and cl, dl						;rcx = w_size mod 8
	jz .multiple_of_8

	;Caso no-multiplo: Notar que este ciclo tiene como mucho 7 iteraciones
	.not_multiple_of_8:
		movq xmm2, [rsi]		;xmm2 = nw_0
		movq xmm1, [rdi]		;xmm1 = w_0
		mulsd xmm2, xmm0		;xmm2 = c * nw_0
		subsd xmm1, xmm2		;xmm1 = w_0 - c * nw_0
		add rsi, 8
		movq [rdi], xmm1
		add rdi, 8
		dec rcx
		jnz .not_multiple_of_8
		and dl, 0xF8

	;Inicializo el contador
	.multiple_of_8:
		;Quiero determinar si la cantidad de iteraciones es par
		vbroadcastsd ymm0, xmm0	;ymm0 = c | c | c | c

		;Itero sobre todos los pesos y realizo la actualizacion
		.ciclo:
			%rep UNROLL_AMT
			vmovupd ymm2, [rsi]	;xmm2 = | nw_i| nw_i+1|
			vmovupd ymm1, [rdi]	;xmm1 = | w_i | w_i+1 |

			vmulpd ymm2, ymm0
			vsubpd ymm1, ymm2
			add rsi, 32
			vmovupd [rdi], ymm1

			;Avanzo los punteros
			add rdi, 32
			%endrep
			sub rdx, 8
			jnz .ciclo

		pop rbp
	  ret

 ;  matrix_prod:
	; push rbp
	; mov rbp, rsp
	; push r12
	; push r13
	; push r14
	; push r15

	; ;Salvo las dimensiones 
	; mov r10, rdx ; r10 = n
	; ;lea r11, [rcx - 1] ; r11 = m
	; ;lea r12, [r8 - 1]  ; r12 = l
	; ;mov r13, rdx ; En r13 guardo la cte n (en rdx) pues voy a necesitar rdx para multiplicar
	; ;xor r10, r10
	; ;La cantidad de iteraciones internas es m = rcx

	; mov r10, rdx ; r10 = n

	; ;Calculo principio de la ultima fila de matrix2
	; lea rax, [rcx - 1]
	; mul r8
	; shl rdx, 32
	; lea r13, [rdx + rax] ; r13 = i * l

	; .i:
	; 	;xor r12, r12
	; 	mov r12, r8  ; r12 = l

	; 	; Calculo desplazamiento en matrix1
	; 	mov rax, r10
	; 	mul ecx
	; 	shl rdx, 32
	; 	;mov edx, eax ; rdx = i * m
	; 	lea r14, [rdx + rax] ; r14 = i * m
		
	; 	.j:
	; 		;xor r11, r11
	; 		mov r11, rcx ; r11 = m
	; 		pxor xmm3, xmm3													;xmm3 = acumulador para el coef r10, r12

	; 		; Calculo desplazamiento en matrix2
	; 		;mov r15, r12

	; 		lea r15, [r13 + r12] 	; Ultima fila, Columna r12 de matrix2
	; 		.k:
	; 			movq xmm1, [rdi + 8 * r14]		;xmm1 = matrix1[r10][r11]
	; 			movq xmm2, [rsi + 8 * r15]		;xmm2 = matrix2[r11][r12]
	; 			mulsd xmm1, xmm2
	; 			addsd xmm3, xmm1
				
	; 			dec r14 			;Voy del primer al ultimo elemento de la fila
	; 			sub r15, r8 	; Retrocedo una fila
	; 			dec r11
	; 			;cmp r11, rcx
	; 			jnz .k
	; 		add r14, rcx

	; 		; Calculo desplazamiento en output
	; 		lea rax, [r10 - 1]
	; 		mul r8d
	; 		shl rdx, 32
	; 		;mov edx, eax ; rdx = i * m
	; 		lea r15, [rdx + rax]
	; 		add r15, r12
			
	; 		movq [r9 + 8 * r15], xmm3
	; 		dec r12
	; 		;cmp r12, r8
	; 		jnz .j
	; 	dec r10
	; 	;cmp r10, r13
	; 	jnz .i

	; pop r15
	; pop r14
	; pop r13
	; pop r12
	; pop rbp
	; ret


;;;;;;;;;;;;;;;;;STOP;;;;;;;;;;;;;;;;;;;;;;;
	;void mat_plus_vec(
	;	double* matrix, (rdi) 
	; double* vector, (rsi)
	; uint n, 				(rdx)
	; uint m, 				(rcx)
	; double* output 	(r8)
	; )
	; mat_plus_vec:
	; push rbp
	; mov rbp, rsp

	; ;Calculo la cantidad de elementos total
	; xor rax, rax
	; mov eax, edx
	; mul ecx					;eax = low(n*m) ;edx = high(n*m)
	; shl rdx, 32
	; add rax, rdx			;rax = #elementos

	; ;Chequeo si la cantidad de elementos es multiplo de 4
	; mov rdx, 3
	; and dx, ax
	; jz .multiple_of_4

	; .not_multiple_of_4:
	; ;Caso impar: opero sobre el primer elemento por separado
	; movd xmm1, [rdi]
	; movd xmm2, [rsi]
	; addsd xmm1, xmm2
	; movd [r8], xmm1
	; add rdi, 8
	; add rsi, 8
	; add r8, 8	
	; dec rdx
	; jnz .not_multiple_of_4
	; and al, 0xFC		;Seteo en 0 los dos ultimos bits de rax

	; ;Inicializo el contador
	; .multiple_of_4:
	; ;shr rax, 2				;Proceso de a 4 elementos 

	; ;Itero sobre todos los pixeles y realizo la operación de SUBPD
	; .ciclo:
	; 	vmovupd ymm1, [rdi]	;xmm1 = | px0 | px1 |
	; 	vmovupd ymm2, [rsi]	;xmm2 = | px0'| px1'|

	; 	vaddpd ymm1, ymm2

	; 	vmovupd [r8], ymm1

	; 	;Avanzo los punteros
	; 	add rdi, 32
	; 	add rsi, 32
	; 	add r8, 32
	; 	sub rax, 4
	; 	jnz .ciclo
	
	; pop rbp
 ;  ret

; inputs: rdi, rsi, rdx, rcx, r8, r9


;int max_arg(
;	double* vector, (rdi) 
; uint n 					(rsi)
;)
;	max_arg:
;	push rbp
;	mov rbp, rsp


;	pop rbp
;	ret

;;;;;;;;;;;;;; Version XMM de mat_plus_vec ;;;;;;;;;;;;;;;;;;;
; mat_plus_vec:
; 	push rbp
; 	mov rbp, rsp

; 	;Calculo la cantidad de pixeles total
; 	xor rax, rax
; 	mov eax, edx
; 	mul ecx					;eax = low(n*m) ;edx = high(n*m)
; 	shl rdx, 32
; 	add rax, rdx			;rax = #elementos

; 	;Chequeo si la cantidad de elementos es par
; 	xor rdx, rdx
; 	inc rdx
; 	and dl, al
; 	cmp dl, 0
; 	jz .even_case

; 	.odd_case:
; 	;Caso impar: opero sobre el primer elemento por separado
; 	movd xmm1, [rdi]
; 	movd xmm2, [rsi]
; 	addsd xmm1, xmm2
; 	movd [r8], xmm1
; 	add rdi, 8
; 	add rsi, 8
; 	add r8, 8	
; 	;dec rax

; 	;Inicializo el contador
; 	.even_case:
; 	mov rcx, rax
; 	shr rcx, 1				;Proceso de a 2 pixeles, 

; 	;Itero sobre todos los pixeles y realizo la operación de SUBPD
; 	.ciclo:
; 		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 |
; 		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'|

; 		addpd xmm1, xmm2

; 		movupd [r8], xmm1

; 		;Avanzo los punteros
; 		add rdi, 16
; 		add rsi, 16
; 		add r8, 16
; 		loop .ciclo
	
; 	pop rbp
;    	ret

; update_weight:
; 	push rbp
; 	mov rbp, rsp

; 	;Calculo w_size mod 4
; 	xor rcx, rcx
; 	inc cl
; 	and cl, dl						;rcx = w_size mod 4
; 	cmp cl, 0
; 	jz .multiple_of_4

; 	;Caso no-multiplo
; 	.not_multiple_of_4:
; 		movd xmm1, [rdi]		;xmm1 = w_0
; 		movd xmm2, [rsi]		;xmm2 = nw_0
; 		mulsd xmm2, xmm0		;xmm2 = c * nw_0
; 		subsd xmm1, xmm2		;xmm1 = w_0 - c * nw_0
; 		movd [rdi], xmm1
; 		add rdi, 8
; 		add rsi, 8
; 		;loop .not_multiple_of_4
; 		;dec rcx 						;rcx = w_size - 1

; 	;Inicializo el contador
; 	.multiple_of_4:
; 	mov rcx, rdx 						;rcx = w_size
; 	shr rcx, 1						;Proceso de a 4 elementos
; 	unpcklpd xmm0, xmm0

; 	;Itero sobre todos los pesos y realizo la actualizacion
; 	.ciclo:
; 		movupd xmm1, [rdi]	;xmm1 = | w_i | w_i+1 |
; 		movupd xmm2, [rsi]	;xmm2 = | nw_i| nw_i+1|

; 		mulpd xmm2, xmm0
; 		subpd xmm1, xmm2
; 		movupd [rdi], xmm1

; 		;Avanzo los punteros
; 		add rdi, 16
; 		add rsi, 16
; 		loop .ciclo
	
; 	pop rbp
;    	ret

;void matrix_prod(
;	double* matrix1, (rdi) 
;	double* matrix2, (rsi)
; uint n, 				 (rdx)				
;	uint m, 				 (rcx)
; uint l, 				 (r8)	
; double* output   (r9)
;)

;Version de atras para adelante

matrix_prod:
	push rbp
	mov rbp, rsp
	push r12
	push r13
	push r14
	push r15

	mov r10, rdx ; r10 = n

	; Calculo desplazamiento del ultimo elemento
	; de matrix1
	mov rax, r10
	mul ecx
	shl rdx, 32
	;mov edx, eax ; rdx = i * m
	lea r14, [rdx + rax - 1] ; r14 = i * m

	;Precomputo el offset del ultimo elemento de la anteultima fila de matrix2
	lea rax, [rcx - 1]
	mul r8d
	shl rdx, 32
	;mov edx, eax ; rdx = i * m
	lea r13, [rdx + rax - 1]

	.i:
		mov r12, r8
		.j:
			mov r11, rcx		;Uso r11 como contador unicamente
			
			pxor xmm3, xmm3													;xmm3 = acumulador para el coef r10, r12

			; Calculo desplazamiento en matrix2
			lea r15, [r13 + r12]

			.k:
				movq xmm1, [rdi + 8 * r14]		;xmm1 = matrix1[r10][r11]
				movq xmm2, [rsi + 8 * r15]		;xmm2 = matrix2[r11][r12]
				mulsd xmm1, xmm2
				addsd xmm3, xmm1
				sub r15, r8 ;Voy del ultimo al primer elemento de la columna
				dec r14 		;Voy del ultimo al primer elemento de la fila
				dec r11
				jnz .k
			add r14, rcx	;Hago esto para situarme de vuelta
										;al final de la fila r10-1
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

	pop r15
	pop r14
	pop r13
	pop r12
	pop rbp
	ret
