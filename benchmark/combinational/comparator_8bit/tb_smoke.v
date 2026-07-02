`timescale 1ns/1ps
module tb_smoke;
    reg  [7:0] a, b;
    wire eq, gt, lt;
    integer i, j;
    reg [7:0] corners [0:5];

    comparator_8bit dut (.a(a), .b(b), .eq(eq), .gt(gt), .lt(lt));

    initial begin
        corners[0]=8'h00; corners[1]=8'h01; corners[2]=8'h7F;
        corners[3]=8'h80; corners[4]=8'hFE; corners[5]=8'hFF;
        for (i = 0; i < 6; i = i + 1)
            for (j = 0; j < 6; j = j + 1) begin
                a = corners[i]; b = corners[j];
                #1 $display("a=%h b=%h eq=%b gt=%b lt=%b", a, b, eq, gt, lt);
            end
        // adjacent-value boundary sweep
        for (i = 0; i < 255; i = i + 8) begin
            a = i[7:0]; b = i[7:0] + 8'h01;
            #1 $display("a=%h b=%h eq=%b gt=%b lt=%b", a, b, eq, gt, lt);
        end
        for (i = 0; i < 150; i = i + 1) begin
            a = $random; b = $random;
            #1 $display("a=%h b=%h eq=%b gt=%b lt=%b", a, b, eq, gt, lt);
        end
        $finish;
    end
endmodule
