// counter_8bit — MUTANT (bug07_load_msb_dropped.v) — parallel load drops d[7]; values >= 0x80 load wrong
module counter_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       load,
    input  wire [7:0] d,
    input  wire       en,
    input  wire       up_down,
    output reg  [7:0] q,
    output reg        overflow
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q        <= 8'h00;
            overflow <= 1'b0;
        end else if (load) begin
            q        <= {1'b0, d[6:0]};  // BUG: MSB dropped
            overflow <= 1'b0;
        end else if (en) begin
            if (up_down) begin
                q        <= q + 8'h01;
                overflow <= (q == 8'hFF);
            end else begin
                q        <= q - 8'h01;
                overflow <= (q == 8'h00);
            end
        end else begin
            overflow <= 1'b0;
        end
    end
endmodule
