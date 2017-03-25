; RECORDATORIOS
; inputs: rdi, rsi, rdx, rcx, r8, r9
; preservar: r12, r13, r14, r15, rbx, 
; la pila: rbp, rsp
; devolver cosas por rax o xmmo 
; inputs floats: xmm0, xmm1, ..., xmm7

	global cost_derivative_asm_double
	global mat_plus_vec_asm_double
	global update_weight_asm_double
	global hadamardProduct_asm_double
	global cost_derivative_asm_float
	global mat_plus_vec_asm_float
	global update_weight_asm_float
	global hadamardProduct_asm_float

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
	
	%define LF			10


section .rodata
	UNROLL_AMT  equ   2   ; # of times to unroll the loop


section .data
	msg1: DB '%s', LF, 0	;imprimo string con salto de linea y fin de linea
	msg2: DB 'a',0			;modo append para fopen
	msg3: DB '<oracionVacia>',0
	

section .text


;//////////////// FUNCIONES MATRICIALES ////////////////;

;;;;;;;;;;;;;;;;; IMPLEMENTACION DOUBLE ;;;;;;;;;;;;;;;;;

; void cost_derivative(
;	double* res_vec  (rdi)
;	double* target_vec (rsi)
;	double* output	(rdx)
; )
;NOTA: cost_derivative es bastante mas eficiente con SSE2 que con AVR

  cost_derivative_asm_double:
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

  mat_plus_vec_asm_double:
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
	movd xmm1, [rdi]
	movd xmm2, [rsi]
	addsd xmm1, xmm2
	movd [r8], xmm1
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

		addpd xmm1, xmm2

		movupd [r8], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add r8, 16
		sub rax, 2
		jnz .ciclo
	
	pop rbp
  ret


  hadamardProduct_asm_double:
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
	movd xmm1, [rdi]
	movd xmm2, [rsi]

	mulpd xmm1, xmm2

	movd [r8], xmm1
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

  update_weight_asm_double:
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
		movd xmm1, [rdi]		;xmm1 = w_0
		movd xmm2, [rsi]		;xmm2 = nw_0
		mulsd xmm2, xmm0		;xmm2 = c * nw_0
		subsd xmm1, xmm2		;xmm1 = w_0 - c * nw_0
		movd [rdi], xmm1
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


;;;;;;;;;;;;;;;;; IMPLEMENTACION FLOAT ;;;;;;;;;;;;;;;;;

; void cost_derivative(
;	double* res_vec  (rdi)
;	double* target_vec (rsi)
;	double* output	(rdx)
; )

;NOTA: cost_derivative es bastante mas eficiente con SSE2 que con AVR

  cost_derivative_asm_float:
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
		movupd xmm1, [rdi]	;xmm1 = | x0 | x1 | x2 | x3 |
		movupd xmm2, [rsi]	;xmm2 = | y0 | y1 | y2 | y3 |

		subps xmm1, xmm2

		movupd [rdx], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add rdx, 16
	%endrep

	movupd xmm1, [rdi]	;xmm1 = | x0 | x1 |
	movupd xmm2, [rsi]	;xmm2 = | y0 | y1 |

	subps xmm1, xmm2

	movupd [rdx], xmm1

	pop rbp
  ret

  mat_plus_vec_asm_float:
	push rbp
	mov rbp, rsp

	;Calculo la cantidad de pixeles total
	xor rax, rax
	mov eax, edx
	mul ecx					;eax = low(n*m) ;edx = high(n*m)
	shl rdx, 32
	add rax, rdx			;rax = #elementos

	;Chequeo si la cantidad de elementos es par
	xor rdx, rdx
	inc rdx
	and rdx, rax
	jz .A

	.B:
	;Caso impar: opero sobre el primer elemento por separado
	movd xmm1, [rdi]
	movd xmm2, [rsi]
	addss xmm1, xmm2
	movd [r8], xmm1
	add rdi, 4
	add rsi, 4
	add r8, 4	
	dec rdx
	jnz .B

	;Inicializo el contador
	.A:
	and al, 0xFE
	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'|

		addps xmm1, xmm2

		movupd [r8], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		add r8, 16
		sub rax, 4
		jnz .ciclo
	
	pop rbp
  ret


  hadamardProduct_asm_float:
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
	and al, 0xFE
	;Itero sobre todos los pixeles y realizo la operación de SUBPD
	.ciclo:
		movupd xmm1, [rdi]	;xmm1 = | px0 | px1 |
		movupd xmm2, [rsi]	;xmm2 = | px0'| px1'|

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

  update_weight_asm_float:
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
	mov rcx, rdx 						;rcx = w_size
	shr rcx, 2						;Proceso de a 4 elementos
	unpcklps xmm0, xmm0

	;Itero sobre todos los pesos y realizo la actualizacion
	.ciclo:
		movups xmm1, [rdi]	;xmm1 = | w_i | w_i+1 |
		movups xmm2, [rsi]	;xmm2 = | nw_i| nw_i+1|

		mulps xmm2, xmm0
		subps xmm1, xmm2
		movups [rdi], xmm1

		;Avanzo los punteros
		add rdi, 16
		add rsi, 16
		loop .ciclo
	
  	pop rbp
  ret

;;;;;;;;;;;;;;;;;COSAS COMENTADAS;;;;;;;;;;;;;;;;;;;;;;;

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


;void update_weight(
; 		double* w, 		(rdi) 
;			double* nw, 	(rsi)
;			uint w_size, 	(rdx)	
;			double c    	(xmm0)
;)
	; update_weight:
	; push rbp
	; mov rbp, rsp

	; ;Calculo w_size mod 8
	; mov rcx, 3
	; and cl, dl						;rcx = w_size mod 8
	; jz .multiple_of_8

	; ;Caso no-multiplo: Notar que este ciclo tiene como mucho 7 iteraciones
	; .not_multiple_of_8:
	; 	movd xmm2, [rsi]		;xmm2 = nw_0
	; 	movd xmm1, [rdi]		;xmm1 = w_0
	; 	mulsd xmm2, xmm0		;xmm2 = c * nw_0
	; 	subsd xmm1, xmm2		;xmm1 = w_0 - c * nw_0
	; 	add rsi, 8
	; 	movd [rdi], xmm1
	; 	add rdi, 8
	; 	dec rcx
	; 	jnz .not_multiple_of_8
	; 	and dl, 0xFC

	; ;Inicializo el contador
	; .multiple_of_8:
	; 	;Quiero determinar si la cantidad de iteraciones es par
	; 	;vbroadcastsd ymm0, xmm0	;ymm0 = c | c | c | c
	; 	unpcklpd xmm0, xmm0

	; 	;Itero sobre todos los pesos y realizo la actualizacion
	; 	.ciclo:
	; 		%rep UNROLL_AMT
	; 		movupd xmm2, [rsi]	;xmm2 = | nw_i| nw_i+1|
	; 		movupd xmm1, [rdi]	;xmm1 = | w_i | w_i+1 |

	; 		mulpd xmm2, xmm0
	; 		subpd xmm1, xmm2
	; 		add rsi, 16
	; 		movupd [rdi], xmm1

	; 		;Avanzo los punteros
	; 		add rdi, 16
	; 		%endrep
	; 		sub rdx, 8
	; 		jnz .ciclo

	; 	pop rbp
	;   ret


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

