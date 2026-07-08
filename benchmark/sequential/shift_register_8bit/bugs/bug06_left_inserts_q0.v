// shift_register_8bit — MUTANT (bug06_left_inserts_q0.v) — left shift recycles q[0] instead of taking serial_in
module shift_register_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       load,
    input  wire [7:0] d,
    input  wire       shift_en,
    input  wire       dir,
    input  wire       serial_in,
    output reg  [7:0] q,
    output wire       serial_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 8'h00;
        else if (load)
            q <= d;
        else if (shift_en) begin
            if (dir == 1'b0)
                q <= {q[6:0], q[0]};   // BUG: recycles LSB
            else
                q <= {serial_in, q[7:1]};   // shift right
        end
    end
    assign serial_out = (dir == 1'b0) ? q[7] : q[0];
endmodule
