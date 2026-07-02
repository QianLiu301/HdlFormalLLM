`timescale 1ns/1ps
module tb_smoke;
    reg clk, rst_n, wr_en, rd_en;
    reg [7:0] din;
    wire [7:0] dout;
    wire full, empty;
    integer i;

    sync_fifo_8x8 dut (.clk(clk), .rst_n(rst_n), .wr_en(wr_en), .din(din),
                       .rd_en(rd_en), .dout(dout), .full(full), .empty(empty));

    always #5 clk = ~clk;

    task show; begin
        $display("t=%0t w=%b r=%b din=%h dout=%h f=%b e=%b", $time, wr_en, rd_en, din, dout, full, empty);
    end endtask

    initial begin
        clk = 0; rst_n = 0; wr_en = 0; rd_en = 0; din = 8'h00;
        @(negedge clk) show; rst_n = 1;
        // read from empty (ignored, dout holds)
        rd_en = 1; @(negedge clk) show; rd_en = 0;
        // fill with 10 writes (9th/10th must be dropped)
        wr_en = 1;
        for (i = 0; i < 10; i = i + 1) begin
            din = 8'h10 + i[7:0]; @(negedge clk) show;
        end
        wr_en = 0;
        // write while full + simultaneous read (write must be dropped)
        wr_en = 1; rd_en = 1; din = 8'hEE; @(negedge clk) show;
        wr_en = 0; rd_en = 0;
        // drain everything (order check) + extra reads past empty
        rd_en = 1;
        for (i = 0; i < 10; i = i + 1) begin @(negedge clk) show; end
        rd_en = 0;
        // simultaneous rd+wr at intermediate occupancy
        wr_en = 1; din = 8'hA1; @(negedge clk) show;
        din = 8'hA2; @(negedge clk) show;
        rd_en = 1;  din = 8'hA3; @(negedge clk) show;
        din = 8'hA4; @(negedge clk) show;
        wr_en = 0; @(negedge clk) show; @(negedge clk) show;
        rd_en = 0;
        // pseudo-random traffic
        for (i = 0; i < 60; i = i + 1) begin
            {wr_en, rd_en} = $random; din = $random;
            @(negedge clk) show;
        end
        $finish;
    end
endmodule
