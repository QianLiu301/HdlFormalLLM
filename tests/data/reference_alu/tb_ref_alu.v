// Reads "op a b" (hex) per line from vectors.txt and prints the outputs of ref_alu.
module tb_ref_alu;
    parameter W = 8;
    reg  [3:0]   op;
    reg  [W-1:0] a, b;
    wire [W-1:0] y;
    wire zero, negative, ovf_signed, ovf_carry;
    integer fd, n;
    ref_alu #(.W(W)) dut (.op(op), .a(a), .b(b), .y(y), .zero(zero), .negative(negative),
                          .ovf_signed(ovf_signed), .ovf_carry(ovf_carry));
    initial begin
        fd = $fopen("vectors.txt", "r");
        n = $fscanf(fd, "%h %h %h\n", op, a, b);
        while (n == 3) begin
            #1 $display("%0h %0h %0h %0h %0d %0d %0d %0d", op, a, b, y, zero, negative, ovf_signed, ovf_carry);
            n = $fscanf(fd, "%h %h %h\n", op, a, b);
        end
        $finish;
    end
endmodule
