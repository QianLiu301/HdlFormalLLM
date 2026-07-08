`timescale 1ns/1ps
module tb_smoke;
    reg clk, rst_n, we;
    reg [3:0] waddr, raddr_a, raddr_b;
    reg [7:0] wdata;
    wire [7:0] rdata_a, rdata_b;
    integer i;

    register_file_16x8 dut (.clk(clk), .rst_n(rst_n), .we(we),
                            .waddr(waddr), .wdata(wdata),
                            .raddr_a(raddr_a), .raddr_b(raddr_b),
                            .rdata_a(rdata_a), .rdata_b(rdata_b));

    always #5 clk = ~clk;

    task show; begin
        $display("t=%0t we=%b wa=%h wd=%h ra=%h rb=%h da=%h db=%h", $time, we, waddr, wdata, raddr_a, raddr_b, rdata_a, rdata_b);
    end endtask

    initial begin
        clk = 0; rst_n = 0; we = 0; waddr = 0; wdata = 0; raddr_a = 0; raddr_b = 0;
        @(negedge clk) show; rst_n = 1;
        // write a distinct value to every address (incl. attempt on R0)
        we = 1;
        for (i = 0; i < 16; i = i + 1) begin
            waddr = i[3:0]; wdata = 8'hC0 + i[7:0];
            @(negedge clk) show;
        end
        we = 0;
        // read back all addresses via both ports (aliasing + R0 check)
        for (i = 0; i < 16; i = i + 1) begin
            raddr_a = i[3:0]; raddr_b = 4'd15 - i[3:0];
            #1 show;
        end
        // write gating: we=0 must not write
        waddr = 4'd5; wdata = 8'hDE; we = 0; @(negedge clk);
        raddr_a = 4'd5; #1 show;
        // read-during-write: old value visible before edge
        we = 1; waddr = 4'd7; wdata = 8'h99; raddr_a = 4'd7;
        #1 show;             // before edge: old value
        @(negedge clk) show; // after edge: new value
        we = 0;
        // pseudo-random access
        for (i = 0; i < 40; i = i + 1) begin
            we = $random; waddr = $random; wdata = $random;
            raddr_a = $random; raddr_b = $random;
            @(negedge clk) show;
        end
        // second reset with registers already written: all 16 must read 0x00
        we = 0; rst_n = 0; @(negedge clk) show; rst_n = 1;
        for (i = 0; i < 16; i = i + 1) begin
            raddr_a = i[3:0]; raddr_b = 4'd15 - i[3:0];
            #1 show;
        end
        $finish;
    end
endmodule
